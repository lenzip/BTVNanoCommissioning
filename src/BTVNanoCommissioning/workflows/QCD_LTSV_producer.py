import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights
import vector
vector.register_awkward()

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    common_shifts,
    weight_manager,
)

# user helper function
from BTVNanoCommissioning.helpers.func import (
    flatten,
    update,
    uproot_writeable,
    dump_lumi,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogrammer import histogrammer, histo_writter
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    MET_filters
)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        events = missing_branch(events)
        shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )


    ## Processed events per-chunk, made selections, filled histogram, stored root files
    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        ######################
        #  Create histogram  # : Get the histogram dict from `histogrammer`
        ######################
        # this is the place to modify
        output = {} if self.noHist else histogrammer(events, "example")

        if shift_name is None:
            if isRealData:
                output["sumw"] = len(events)
            else:
                output["sumw"] = ak.sum(events.genWeight)

        ####################
        #    Selections    #
        ####################
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        met_filter = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
            met_filter = MET_filters(events, campaign)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)
        ##====> start here, make your customize modification
        ## HLT
        triggers = [
            "BTagMu_AK4DiJet20_Mu5",
            "BTagMu_AK4DiJet40_Mu5",
            "BTagMu_AK4DiJet70_Mu5",
            "BTagMu_AK4DiJet110_Mu5",
            "BTagMu_AK4DiJet170_Mu5",
            "BTagMu_AK4Jet300_Mu5",
        ]
        req_trig = HLT_helper(events, triggers)

        ##### Add some selections
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2

        muon_sel = (events.Muon.pt > 5) & (mu_idiso(events, self._campaign))
        event_mu = events.Muon[muon_sel]
        req_muon = ak.num(event_mu.pt) >= 1

        req_leadlep_pt = ak.any(
            event_mu.pt > 5, axis=-1
        )

        ## Jet cuts
        jet_sel = (events.Jet.pt > 20) & (abs(events.Jet.eta)<2.4) & (jet_id(events, self._campaign))
        event_jet = events.Jet[jet_sel]
        
        
        ## Other cuts
        muon_jet_pairs = ak.cartesian({'mu': event_mu, 'jet': event_jet}, axis=1, nested=True)
        dR = muon_jet_pairs['jet'].delta_r(muon_jet_pairs['mu'])
        matching_pairs=muon_jet_pairs[dR<0.4]
        matching_jets = ak.flatten(matching_pairs['jet'], axis=2)
        req_jet = ak.num(matching_jets) >= 1

        pt_matrix = {'20': {'ptmin': 20,   'ptmax': 50,      'trigger': 'BTagMu_AK4DiJet20_Mu5',  'req': 2, 'req_matching': 1},
                     '50': {'ptmin': 50,   'ptmax': 80,      'trigger': 'BTagMu_AK4DiJet40_Mu5',  'req': 2, 'req_matching': 1},
                     '80': {'ptmin': 80,   'ptmax': 120,     'trigger': 'BTagMu_AK4DiJet70_Mu5',  'req': 2, 'req_matching': 1},
                     '120': {'ptmin': 120, 'ptmax': 180,     'trigger': 'BTagMu_AK4DiJet110_Mu5', 'req': 2, 'req_matching': 1},
                     '180': {'ptmin': 180, 'ptmax': 320,     'trigger': 'BTagMu_AK4DiJet170_Mu5', 'req': 2, 'req_matching': 1},
                     '320': {'ptmin': 320, 'ptmax': 9999999, 'trigger': 'BTagMu_AK4Jet300_Mu5',   'req': 1, 'req_matching': 1}}
        # Build selection mask
        masks = []

        for bin_key, cfg in pt_matrix.items():
            # select jets in this pt range
            jets_in_bin  = (event_jet.pt >= cfg['ptmin']) & (event_jet.pt < cfg['ptmax'])
            mjets_in_bin = (matching_jets.pt >= cfg['ptmin']) & (matching_jets.pt < cfg['ptmax'])

            # count how many jets pass the pt range
            n_jets_in_bin = ak.num(event_jet[jets_in_bin])
            n_mjets_in_bin = ak.num(matching_jets[mjets_in_bin])

            # build event mask: trigger fired AND enough jets
            mask = (events.HLT[cfg['trigger']]) & (n_jets_in_bin >= cfg['req']) & (n_mjets_in_bin >= cfg['req_matching'])
            masks.append(mask)
        
        trigger_pt_mask_selection = np.logical_or.reduce(masks)
        

        ## Apply all selections
        event_level = (
            req_trig & req_lumi & met_filter & req_jet & req_muon & req_leadlep_pt & trigger_pt_mask_selection
        )

        ##<==== finish selection
        event_level = ak.fill_none(event_level, False)
        # Skip empty events -
        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}
        ##===>  Ntuplization  : store custom information
        ####################
        # Selected objects # : Pruned objects with reduced event_level
        ####################
        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = matching_jets[event_level][:, 0]
        pruned_ev['njet'] = ak.num(matching_jets[event_level])
        print(f'passed {len(pruned_ev)}')

        ## <========= end: store custom objects
        ####################
        #     Output       #
        ####################
        # Configure SFs - read pruned objects from the pruned_ev and apply SFs and call the systematics
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
        # Configure systematics shifts
        if shift_name is None:
            systematics = ["nominal"] + list(
                weights.variations
            )  # nominal + weight variation systematics
        else:
            systematics = [shift_name]  # JES/JER systematics

        # Fill the weight to output arrys

        # Configure histograms- fill the histograms with pruned objects
        if not self.noHist:
            output = histo_writter(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # Output arrays - store the pruned objects in the output arrays
        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )

        return {dataset: output}

    ## post process, return the accumulator, compressed
    def postprocess(self, accumulator):
        return accumulator
