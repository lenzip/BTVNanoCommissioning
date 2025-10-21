import collections, awkward as ak, numpy as np
import os
import uproot
import hist
from coffea import processor
from coffea.analysis_tools import Weights
import correctionlib
import vector

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
#from BTVNanoCommissioning.utils.histogrammer import histogrammer, histo_writter
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    MET_filters,
    btag_wp_dict,
)

# Histogram axes definition
CHANNELS = ["incl"]
HISTOGRAM_AXES = {
    "flav_axis": hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Flavor"),
    "sys_axis": hist.axis.StrCategory([], name="syst", growth=True),
    "njet_axis": hist.axis.Integer(0, 10, name="njet", label="N jets"),
    "dr_axis": hist.axis.Regular(20, 0, 8, name="dr", label="$\Delta$R"),
    "pt_axis": hist.axis.Regular(50, 0, 300, name="pt", label=r"$p_{\mathrm{T}}$ / GeV"),
    "ptbin_axis": hist.axis.StrCategory([], name="ptbin", label=r"$p_\mathrm{T}$ bin", growth=True),
    "pt_btag_axis": hist.axis.Variable([*range(0, 200, 10), 200, np.inf], name="pt", label=r"$p_{\mathrm{T}}$ / GeV"),
    "mass_axis": hist.axis.Regular(50, 0, 300, name="mass", label="$m$ / GeV"),
    "eta_axis": hist.axis.Regular(25, -2.5, 2.5, name="eta", label="$\eta$"),
    "abs_eta_axis": hist.axis.Variable([0.0, 0.8, 1.6, 2.5], name="eta", label="|$\eta$|"),
    "phi_axis": hist.axis.Regular(30, -3, 3, name="phi", label="$\phi$"),
    "Proba_axis": hist.axis.Regular(20, 0, 2, name="Proba", label="JP"),
    "DeepJet_sv_mass_0_axis": hist.axis.Regular(20, 0, 10, name="DeepJet_sv_mass_0", label=r"$m_\mathrm{SV}$ / GeV"),
}


# Define histograms
def define_histograms(
    particle_objects: dict[str, list],
    tagger_list: list,
):
    histograms = {
        "njet": hist.Hist(HISTOGRAM_AXES["sys_axis"], HISTOGRAM_AXES["njet_axis"], hist.storage.Weight()),
        "dr_jets": hist.Hist(HISTOGRAM_AXES["sys_axis"], HISTOGRAM_AXES["dr_axis"], hist.storage.Weight()),
    }

    for obj, attrs in particle_objects.items():
        for attr in attrs:
            if obj == "SelJet":
                for tagger in tagger_list:
                    # syst, flav, ptbin, attr
                    HISTOGRAM_AXES[f"{tagger}_btagwp_axis"] = hist.axis.IntCategory([0, 1, 2, 3, 4, 5], name=f"{tagger}_btagwp", label=f"{tagger} WP passed")
                    histograms[f"{obj}_{tagger}_{attr}"] = hist.Hist(
                        HISTOGRAM_AXES["sys_axis"],
                        HISTOGRAM_AXES["flav_axis"],
                        HISTOGRAM_AXES["ptbin_axis"],
                        HISTOGRAM_AXES[f"{attr}_axis"],
                        HISTOGRAM_AXES[f"{tagger}_btagwp_axis"],
                        hist.storage.Weight(),
                    )
            else:
                histograms[f"{obj}_{attr}"] = hist.Hist(
                    HISTOGRAM_AXES["sys_axis"],
                    HISTOGRAM_AXES[f"{attr}_axis"],
                    hist.storage.Weight(),
                )


    return histograms

# Fill histograms
def fill_histograms(
    histograms,
    pruned_ev,
    particle_objects: dict[str, list],
    jetPtBins,
    weights,
    systematics: list,
    isSyst: bool,
    tagger_list: list,
    btag_wps: dict,
    year,
    campaign,
):
    for syst in systematics:
        if not isSyst and syst != "nominal":
            break

        weight = (
            weights.weight()
            if syst == "nominal" or syst not in list(weights.variations)
            else weights.weight(modifier=syst)
        )

        # Global jet histograms
        histograms["njet"].fill(syst=syst, njet=pruned_ev.njet, weight=weight)
        
        
        # Object histograms
        for obj, attrs in particle_objects.items():
            for attr in attrs:
                parts = obj.split("#", 1)
                _obj = parts[0]
                if len(parts) > 1:
                    index = int(parts[1])
                    attr_value = getattr(pruned_ev[_obj][:, index], attr)
                else:
                    attr_value = getattr(pruned_ev[_obj], attr)


                # flatten if needed
                if attr_value.ndim > 1:
                    weight_flat, attr_value_flat = broadcast_and_flatten(weight, attr_value)
                else:
                    weight_flat = weight
                    attr_value_flat = attr_value


                if obj == "SelJet":
                    flav_flat = ak.flatten(pruned_ev.SelJet.flav, axis=None)
                    jet_pt = pruned_ev.SelJet.pt
                    
                    # Build pt bins
                    ptbins = ak.Array(["undefined"] * len(jet_pt))
                    for name, info in jetPtBins.items():
                        low, high = info["jetPtRange"]
                        mask = (jet_pt >= low) & (jet_pt < high)
                        ptbins = ak.where(mask, name, ptbins)
                    # Flatten pt bins
                    ptbins_flat = ak.to_list(ptbins)


                    for tagger in tagger_list: 
                        bdisc = getattr(pruned_ev.SelJet, f"btag{tagger}B")  # per ora con b vs all
                        wps = btag_wps[tagger]["b"]

                        
                        pass_wp_idx = ak.zeros_like(bdisc, dtype=int)
                        for wp_name, wp_thr in wps.items():
                            if wp_name == "No":
                                continue
                            pass_wp_idx = pass_wp_idx + ak.values_astype(bdisc > float(wp_thr), int)

                        histograms[f"{obj}_{tagger}_{attr}"].fill(
                            syst=syst,
                            ptbin=ptbins_flat,
                            **{attr: attr_value_flat},
                            flav=flav_flat,
                            **{f"{tagger}_btagwp": pass_wp_idx},
                            weight=weight_flat,
                        )
                else:
                    histograms[f"{obj}_{attr}"].fill(**{attr: attr_value_flat, "weight": weight_flat, "syst": syst})


    return histograms

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

        # WP per tagger
        self.btag_wps = btag_wp_dict[self._year + "_" + self._campaign] 


        # pt bins for SF calculation
        ptbins = [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000]
        self.jetPtBins = collections.OrderedDict()
        for i in range(len(ptbins) - 1):
            low = ptbins[i]
            high = ptbins[i + 1]
            name = f"Pt{low}to{high}"
            self.jetPtBins[name] = {}
            self.jetPtBins[name]["jetPtRange"] = [float(low), float(high)]
        


        self.particle_objects = {
            "SelJet": ["pt", "eta", "phi", "mass", "DeepJet_sv_mass_0"],
            "SelMuon": ["pt", "eta", "phi"],  
            "PuppiMET": ["pt", "phi"],        
        }

    @property
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        events = missing_branch(events)
        vetoed_events, shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), name)
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
        #output = {} if self.noHist else histogrammer(events, "example")
        output = {}

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
            met_filter = MET_filters(events, self._campaign)
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
        trigbools = {}
        for bin_key, cfg in pt_matrix.items():
            # select jets in this pt range
            jets_in_bin  = (event_jet.pt >= cfg['ptmin']) & (event_jet.pt < cfg['ptmax'])
            mjets_in_bin = (matching_jets.pt >= cfg['ptmin']) & (matching_jets.pt < cfg['ptmax'])

            # count how many jets pass the pt range
            n_jets_in_bin = ak.num(event_jet[jets_in_bin])
            n_mjets_in_bin = ak.num(matching_jets[mjets_in_bin])

            # build event mask: trigger fired AND enough jets
            mask = (events.HLT[cfg['trigger']]) & (n_jets_in_bin >= cfg['req']) & (n_mjets_in_bin >= cfg['req_matching'])
            trigbools[cfg['trigger']] = HLT_helper(events, [cfg['trigger']])
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
        #print(pruned_ev.fields)
 
        # Leading matched jet
        if ak.any(ak.num(matching_jets[event_level]) > 0):
            leading_jet = matching_jets[event_level][:, 0]
            pruned_ev["SelJet"] = leading_jet
            # print(leading_jet.fields)

            # Flavour tagging
            if "hadronFlavour" in leading_jet.fields:
                isRealData = False
                genflavor = ak.values_astype(
                    leading_jet.hadronFlavour
                    + 1 * (
                        (leading_jet.partonFlavour == 0)
                        & (leading_jet.hadronFlavour == 0)
                    ),
                    int,
                )
                if "MuonJet" in pruned_ev.fields:
                    smflav = ak.values_astype(
                        1 * (
                            (pruned_ev.MuonJet.partonFlavour == 0)
                            & (pruned_ev.MuonJet.hadronFlavour == 0)
                        ) + pruned_ev.MuonJet.hadronFlavour,
                        int,
                    )
            else:
                isRealData = True
                genflavor = ak.zeros_like(leading_jet.pt, dtype=int)
                if "MuonJet" in pruned_ev.fields:
                    smflav = ak.zeros_like(pruned_ev.MuonJet.pt, dtype=int)

            pruned_ev["SelJet"] = ak.with_field(pruned_ev["SelJet"], genflavor, "flav")


        # Leading muon
        if ak.any(ak.num(event_mu[event_level]) > 0):
            pruned_ev["SelMuon"] = event_mu[event_level][:, 0]

        # PuppiMET 
        pruned_ev["PuppiMET"] = events.PuppiMET[event_level]

        # Number of jets
        pruned_ev['njet'] = ak.num(matching_jets[event_level])

        print(f'passed {len(pruned_ev)}')
        

        ## <========= end: store custom objects
        ####################
        #     Output       #
        ####################
        # Configure SFs - read pruned objects from the pruned_ev and apply SFs and call the systematics
        #logger.debug("setting up weight_manager")
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
        if isRealData:
            if self._year == "2022":
                run_num = "355374_362760"
            elif self._year == "2023":
                run_num = "366727_370790"
            elif self._year == "2024":
                run_num = "378985_386951"    

            psweight = np.zeros(len(pruned_ev))
            for trigger, trigbool in trigbools.items():
                psfile = f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{trigger}_run{run_num}.json"
                if not os.path.isfile(psfile):
                    raise NotImplementedError(
                        f"Prescale weights not available for {trigger} in {self._year}. Please run `scripts/dump_prescale.py`."
                    )
                pseval = correctionlib.CorrectionSet.from_file(psfile)
                thispsweight = pseval["prescaleWeight"].evaluate(
                    pruned_ev.run,
                    f"HLT_{trigger}",
                    ak.values_astype(pruned_ev.luminosityBlock, np.float32),
                )
                psweight = ak.where(trigbool[event_level], thispsweight, psweight)
            weights.add("psweight", psweight)
        # Configure systematics shifts
        if shift_name is None:
            systematics = ["nominal"] + list(
                weights.variations
            )  # nominal + weight variation systematics
        else:
            systematics = [shift_name]  # JES/JER systematics

        # Fill the weight to output arrys

        # Configure histograms- fill the histograms with pruned objects
        #if not self.noHist:
        #    output = histo_writter(
        #        pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
        #    )

        # Fill histograms 

        histograms = define_histograms(
            particle_objects=self.particle_objects,
            tagger_list=self.btag_wps.keys(),
        )

        histograms = fill_histograms(
            histograms=histograms,
            pruned_ev=pruned_ev,
            particle_objects=self.particle_objects,
            jetPtBins=self.jetPtBins,
            weights=weights,
            systematics=systematics,
            isSyst=self.isSyst,
            tagger_list=self.btag_wps.keys(),
            btag_wps=self.btag_wps, 
            year=self._year,
            campaign=self._campaign,
        )
        output.update(histograms)


        # Output arrays - store the pruned objects in the output arrays
        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )

        return {dataset: output}

    ## post process, return the accumulator, compressed
    def postprocess(self, accumulator):
        return accumulator
