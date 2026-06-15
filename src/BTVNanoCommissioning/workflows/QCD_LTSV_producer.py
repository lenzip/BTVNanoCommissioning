import collections, awkward as ak, numpy as np
import os
import uproot
import hist
from coffea import processor
from coffea.analysis_tools import Weights
import correctionlib
import vector

from coffea.lookup_tools import extractor
from BTVNanoCommissioning.helpers.BTA_helper import get_hadron_mass

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
            if obj == "MatchedJets":
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


def get_gluonsplitting_weight_from_lastB(lastBHadron, template):
    """
    Gluon splitting (g->bb) weight based on the number of last-B hadrons in the jet.

    If n_lastB >= 2:
      - up   = 1.5
      - down = 0.5
    else:
      - up = down = 1.0

    Parameters
    ----------
    lastBHadron : awkward.Array
        Collection of last B hadrons matched to the jet (per event).
    template : awkward.Array
        Any event-level array used only to build output shapes (e.g. SelJet.pt).

    Returns
    -------
    (nominal, up, down) : awkward.Arrays
        Event-level weights.
    """
    n_lastB = ak.num(lastBHadron)

    nominal = ak.full_like(template, 1.0, dtype=float)
    up      = ak.where(n_lastB >= 2, 1.5, 1.0)
    down    = ak.where(n_lastB >= 2, 0.5, 1.0)

    return (
        ak.values_astype(nominal, float),
        ak.values_astype(up, float),
        ak.values_astype(down, float),
    )

def get_bfragmentation_weight(xB, genJetPt, shift=False):
    """
    Compute b-hadron fragmentation weights and their systematic variations.

    Parameters
    ----------
    xB : awkward.Array
        Fraction of the jet transverse momentum carried by the leading B hadron.
    genJetPt : awkward.Array
        Transverse momentum of the matched generator-level jet.
    shift : bool
        If True, return absolute weights (nominal, up, down).
        If False, return relative variations (1, up/nominal, down/nominal).

    Returns
    -------
    tuple of awkward.Array
        Nominal, up, and down weights (or relative variations).
    """
    ext = extractor()
    ext.add_weight_sets(["* * src/BTVNanoCommissioning/data/BFragmentation/bfragweights_vs_pt.root"])
    ext.finalize()

    passJet = ak.all(xB < 1) & ak.all(genJetPt >= 30)
    failJet = ak.full_like(xB, 1) - passJet

    bfragweight     = ak.values_astype(passJet * (ext.make_evaluator()["fragCP5BL"](xB, genJetPt))     + failJet, float)
    bfragweightUp   = ak.values_astype(passJet * (ext.make_evaluator()["fragCP5BLup"](xB, genJetPt))   + failJet, float)
    bfragweightDown = ak.values_astype(passJet * (ext.make_evaluator()["fragCP5BLdown"](xB, genJetPt)) + failJet, float)

    if shift:
        # Return absolute weights
        return bfragweight, bfragweightUp, bfragweightDown
    else:
        # Return nominal + relative variations
        return ak.full_like(xB, 1.0), bfragweightUp / bfragweight, bfragweightDown / bfragweight

def get_decay_weight(bHadronId, shift=False):
    """
    Compute B-hadron decay (semi-leptonic BR) weights and systematic variations.
    """
    # https://github.com/scodella/BTVNanoCommissioning/blob/008d9b7b17252aa56a6519a180d6775d7d15d37d/src/BTVNanoCommissioning/workflows/pTrel.py#L187
    
    # (constant variations)
    bdecayweight     = ak.full_like(bHadronId, 1.0, dtype=float)
    bdecayweightUp   = ak.full_like(bHadronId, 1.1, dtype=float)
    bdecayweightDown = ak.full_like(bHadronId, 0.9, dtype=float)

    if shift:
        # Return absolute weights
        return bdecayweight, bdecayweightUp, bdecayweightDown
    else:
        # Return nominal + relative variations
        return (
            ak.full_like(bHadronId, 1.0, dtype=float),
            bdecayweightUp / bdecayweight,
            bdecayweightDown / bdecayweight,
        )

def get_cfragmentation_weight(genpart, seljet, dr=0.4):
    """
    c-fragmentation systematic

    Event-by-event logic:
      - consider only charm jets (SelJet.flav == 4)
      - require at least one c-quark within dR < dr from the selected jet
      - select charm hadrons (PDG family 4xx / 4xxx, isLastCopy) within dR < dr
      - if the jet contains:
          D+  (PDG 411) -> Down *= 1.37
          D0  (PDG 421) -> Down *= 0.91
          Ds  (PDG 431) -> Down *= 0.67
      - Up variation is defined as the inverse of Down: Up = 1 / Down

    Returns
    -------
    (nominal, up, down) : awkward.Arrays
        Event-level weights.
    """

    if genpart is None or seljet is None:
        base = seljet.pt if (seljet is not None and hasattr(seljet, "pt")) else ak.Array([])
        ones = ak.full_like(base, 1.0, dtype=float)
        return ones, ones, ones

    # Identify charm jets (hadron flavour == 4)
    if hasattr(seljet, "flav"):
        is_cjet = abs(seljet.flav) == 4
    else:
        # Fallback: if flavour is not available, apply to all jets
        is_cjet = ak.ones_like(seljet.pt, dtype=bool)

    # GenPart information
    abs_pdg = abs(genpart.pdgId)

    # Distance matrix GenPart <-> SelJet
    # Shape: (events, nGenPart, nSelJet=1)
    drs = genpart.metric_table(seljet)
    close_to_jet = drs < dr

    # Require a c-quark inside the jet (|pdgId| == 4)
    cquark_mask = (abs_pdg == 4) & ak.all(close_to_jet, axis=2)
    has_cquark = ak.num(genpart[cquark_mask]) > 0

    # Select charm hadrons (PDG family 4xx / 4xxx), last copy, in the jet
    is_charm_hadron = ((abs_pdg // 100) == 4) | ((abs_pdg // 1000) == 4)

    try:
        is_last = genpart.hasFlags("isLastCopy")
    except Exception:
        # Fallback if hasFlags is not available
        is_last = ak.ones_like(genpart.pdgId, dtype=bool)

    charm_hadrons = genpart[
        is_charm_hadron & is_last & ak.all(close_to_jet, axis=2)
    ]

    abs_ch = abs(charm_hadrons.pdgId)

    has_Dplus = ak.any(abs_ch == 411, axis=-1)
    has_Dzero = ak.any(abs_ch == 421, axis=-1)
    has_Dsubs = ak.any(abs_ch == 431, axis=-1)

    # Build event-level weights
    nominal = ak.full_like(seljet.pt, 1.0, dtype=float)
    down = ak.full_like(seljet.pt, 1.0, dtype=float)

    valid = is_cjet & has_cquark

    down = ak.where(valid & has_Dplus, down * 1.37, down)
    down = ak.where(valid & has_Dzero, down * 0.91, down)
    down = ak.where(valid & has_Dsubs, down * 0.67, down)

    # Up variation defined as inverse of Down (protect against zero)
    safe_down = ak.where(down == 0.0, ak.ones_like(down), down)
    up = 1.0 / safe_down

    return (
        ak.values_astype(nominal, float),
        ak.values_astype(up, float),
        ak.values_astype(down, float),
    )

def get_cdfragmentation_weight(genpart, seljet, dr=0.4, max_steps=15):
    """
    Semi-muonic charm-hadron fragmentation/decay systematic.

    Event-level logic:
      - applied only to b or c jets
      - select D hadrons (411, 421, 431) inside the jet (ΔR < dr)
      - tag a D hadron as semimuonic if a muon descends from it
        (using the GenPart mother chain)
      - apply PDG-based branching-ratio factors to the Down variation
      - Up variation is kept equal to 1

    Returns
    -------
    (nominal, up, down) : awkward.Arrays
        Event-level weights.
    """

    # Safety: return unity weights if inputs are missing
    if genpart is None or seljet is None:
        base = seljet.pt if (seljet is not None and hasattr(seljet, "pt")) else ak.Array([])
        ones = ak.full_like(base, 1.0, dtype=float)
        return ones, ones, ones

    # Require flavour information to identify b/c jets
    if hasattr(seljet, "flav"):
        flav = np.abs(seljet.flav)
        is_bcjet = (flav == 4) | (flav == 5)
    else:
        is_bcjet = ak.zeros_like(seljet.pt, dtype=bool)

    # Distance GenPart <-> selected jet
    close_to_jet = ak.all(genpart.metric_table(seljet) < dr, axis=2)

    abs_pdg = np.abs(genpart.pdgId)

    # Select D hadrons inside the jet
    is_D = (abs_pdg == 411) | (abs_pdg == 421) | (abs_pdg == 431)
    D_in_jet = is_D & close_to_jet

    # Local GenPart indices
    gp_idx = ak.local_index(genpart.pdgId, axis=1)
    D_idx = gp_idx[D_in_jet]

    has_D = ak.num(D_idx) > 0

    # Select muons
    is_mu = abs_pdg == 13
    mu_idx = gp_idx[is_mu]
    has_mu = ak.num(mu_idx) > 0

    mother = genpart.genPartIdxMother

    # Track whether a muon descends from a D hadron of a given species
    anc = mu_idx
    from_Dplus = ak.zeros_like(mu_idx, dtype=bool)
    from_Dzero = ak.zeros_like(mu_idx, dtype=bool)
    from_Dsubs = ak.zeros_like(mu_idx, dtype=bool)

    D_abs = np.abs(genpart.pdgId[D_idx])

    # Helper: check if a muon ancestor index matches ANY D_in_jet index in the same event
    for _ in range(max_steps):
        mom = mother[anc] 
        valid = mom >= 0 #valid mi dice dove mom != -1
        if not ak.any(valid):
            break

        # Compare mom indices with D indices in jet: (event, nMu, nD)
        # matches[event, iMu, iD] = (mom_idx == D_idx_in_jet)
        matches = (mom[..., None] == D_idx[:, None, :]) 

        from_Dplus |= ak.any(matches & (D_abs[:, None, :] == 411), axis=2)
        from_Dzero |= ak.any(matches & (D_abs[:, None, :] == 421), axis=2)
        from_Dsubs |= ak.any(matches & (D_abs[:, None, :] == 431), axis=2)

        anc = ak.where(valid, mom, anc) # where valid == True, anc is updated with mom

    has_DplusMu = ak.any(from_Dplus, axis=1)
    has_DzeroMu = ak.any(from_Dzero, axis=1)
    has_DsubsMu = ak.any(from_Dsubs, axis=1)

    # Build weights
    nominal = ak.full_like(seljet.pt, 1.0, dtype=float)
    up      = ak.full_like(seljet.pt, 1.0, dtype=float)
    down    = ak.full_like(seljet.pt, 1.0, dtype=float)

    valid_evt = is_bcjet & has_D & has_mu

    down = ak.where(valid_evt & has_DplusMu, down * (0.176 / 0.172), down)
    down = ak.where(valid_evt & has_DzeroMu, down * (0.067 / 0.077), down)
    down = ak.where(valid_evt & has_DsubsMu, down * (0.067 / 0.080), down)

    safe_down = ak.where(down == 0.0, ak.ones_like(down), down)
    up = 1.0 / safe_down

    return (
        ak.values_astype(nominal, float),
        ak.values_astype(up, float),
        ak.values_astype(down, float),
    )

def get_cdfragmentation_simple_weight(genpart, seljet, dr=0.4):
    """
    Simplified c->D (D->mu) systematic (no ancestry check).

    Event-by-event logic:
      - apply only to b- or c-jets, if available
      - require at least one charm hadron (D+, D0, Ds) within dR < dr of the selected jet
      - require at least one muon (pdgId==13) within dR < dr of the selected jet
      - if both conditions are met:
          D+  -> Down *= 0.176 / 0.172
          D0  -> Down *= 0.067 / 0.077
          Ds  -> Down *= 0.067 / 0.080
      - Up is defined as the inverse of Down: Up = 1 / Down
      - Nominal is 1

    Returns
    -------
    (nominal, up, down) : awkward.Arrays
        Event-level weights.
    """

    # Base output (event-level)
    ones = ak.full_like(seljet.pt, 1.0, dtype=float)
    nominal = ones
    down = ones

    if genpart is None or seljet is None:
        return nominal, nominal, down

    # Apply only to b/c jets if flavour is available; otherwise apply to all events
    if hasattr(seljet, "flav"):
        flav = np.abs(seljet.flav)
        is_bcjet = (flav == 4) | (flav == 5)
    else:
        is_bcjet = ak.ones_like(seljet.pt, dtype=bool)

    abs_pdg = np.abs(genpart.pdgId)

    # Distance GenPart <-> SelJet (SelJet is 1 jet per event)
    drs = genpart.metric_table(seljet)          # (evt, nGenPart, 1)
    in_jet = ak.all(drs < dr, axis=2)           # (evt, nGenPart)

    # D species inside the jet
    has_Dplus = ak.any(in_jet & (abs_pdg == 411), axis=1)
    has_Dzero = ak.any(in_jet & (abs_pdg == 421), axis=1)
    has_Dsubs = ak.any(in_jet & (abs_pdg == 431), axis=1)

    has_D = has_Dplus | has_Dzero | has_Dsubs

    # Muon inside the jet (no ancestry requirement)
    has_mu_in_jet = ak.any(in_jet & (abs_pdg == 13), axis=1)

    valid = is_bcjet & has_D & has_mu_in_jet

    # Apply the same Down factors as the C++ code (multiplicative)
    down = ak.where(valid & has_Dplus, down * (0.176 / 0.172), down)
    down = ak.where(valid & has_Dzero, down * (0.067 / 0.077), down)
    down = ak.where(valid & has_Dsubs, down * (0.067 / 0.080), down)

    # Up = inverse of Down
    safe_down = ak.where(down == 0.0, ak.ones_like(down), down)
    up = 1.0 / safe_down

    return nominal, up, down


def get_v0_weight(genpart, seljet, dr=0.3):
    """
    V0 (K0s/Lambda) systematic using GenPart.

    Event-level logic:
      - apply only to light jets
      - select GenPart with PDG 310 (K0s) or 3122 (Lambda)
      - require ΔR < dr to selected jet
      - Up   *= 1.3 if at least one K0s
      - Up   *= 1.5 if at least one Lambda
      - Down = 1 / Up  (symmetrized variation)

    Returns
    -------
    (nominal, up, down) : awkward.Arrays
        Event-level weights.
    """

    if genpart is None or seljet is None:
        base = seljet.pt if (seljet is not None and hasattr(seljet, "pt")) else ak.Array([])
        ones = ak.full_like(base, 1.0, dtype=float)
        return ones, ones, ones

    # Apply only to light jets
    if hasattr(seljet, "flav"):
        flav = seljet.flav
        is_light = (flav == 0) | (flav == 1) | (flav == 21)
    else:
        is_light = ak.ones_like(seljet.pt, dtype=bool)

    abs_pdg = np.abs(genpart.pdgId)

    # dR matching
    drs = genpart.metric_table(seljet)
    in_cone = ak.all(drs < dr, axis=2)

    # V0 candidates
    is_k0s = (abs_pdg == 310)
    is_lambda = (abs_pdg == 3122)

    has_k0s = ak.any(in_cone & is_k0s, axis=1)
    has_lambda = ak.any(in_cone & is_lambda, axis=1)

    nominal = ak.full_like(seljet.pt, 1.0, dtype=float)
    up = ak.full_like(seljet.pt, 1.0, dtype=float)

    up = ak.where(is_light & has_k0s, up * 1.3, up)
    up = ak.where(is_light & has_lambda, up * 1.5, up)

    # Symmetric down variation
    safe_up = ak.where(up == 0.0, ak.ones_like(up), up)
    down = 1.0 / safe_up

    return nominal, up, down

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

        # Remove SF mu_Iso from the SF dictionary
        for key in list(self.SF_map.keys()):
            if "mu_Iso" in key:
                self.SF_map.pop(key)

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
            # "SelJet": ["pt", "eta", "phi", "mass", "DeepJet_sv_mass_0", "Proba"],
            "MatchedJets": ["pt", "eta", "phi", "mass", "DeepJet_sv_mass_0", "Proba"],
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

        # --- VETO PROBLEMATIC RUNS ---
        # Remove events belonging to runs that are not covered in the prescale JSON files
        bad_runs = {380126, 380127, 380128, 380238}
        mask_run = ~ak.Array([r in bad_runs for r in events.run])
        events = events[mask_run]


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

        # muon_sel = (events.Muon.pt > 5) & (mu_idiso(events, self._campaign))
        muon_sel = (
                    (events.Muon.pt > 5)
                    & (abs(events.Muon.eta) < 2.4)
                    & (events.Muon.tightId > 0.5)
                    )
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
      
 
        # Leading matched jet
        if ak.any(ak.num(matching_jets[event_level]) > 0):
            leading_jet = matching_jets[event_level][:, 0]
            pruned_ev["SelJet"] = leading_jet
            pruned_ev["MatchedJets"] = matching_jets[event_level] # all matched jets
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

        # ------------------------------------------------------------------
        # Systematic uncertainties (MC only)
        #   - gluonSplitting
        #   - bfragmentation
        #   - Bdecay
        #   - cfragmentation
        #   - cdfragmentation
        #   - V0
        # ------------------------------------------------------------------
        if (not isRealData) and hasattr(pruned_ev, "GenPart") and hasattr(pruned_ev, "GenJet") and hasattr(pruned_ev, "SelJet"):

            # Identify heavy-flavor hadrons using PDG ID encoding
            is_heavy_hadron = lambda p, pid: (abs(p.pdgId) // 100 == pid) | (abs(p.pdgId) // 1000 == pid)

            # Select B hadrons that are final-state copies and geometrically matched to the selected jet
            sel_bhadrons = (
                is_heavy_hadron(pruned_ev.GenPart, 5)
                & pruned_ev.GenPart.hasFlags("isLastCopy")
                & (ak.all(pruned_ev.GenPart.metric_table(pruned_ev.SelJet) < 0.5, axis=2))
            )
            bhadrons = pruned_ev.GenPart[sel_bhadrons]

            # Build a B-hadron collection and remove hadrons with B daughters
            BHadron = ak.zip(
                {
                    "pT": bhadrons.pt,
                    "eta": bhadrons.eta,
                    "phi": bhadrons.phi,
                    "pdgID": bhadrons.pdgId,
                    "mass": get_hadron_mass(bhadrons.pdgId),
                    "hasBdaughter": ak.values_astype(
                        ak.any(is_heavy_hadron(bhadrons.children, 5), axis=-1), int
                    ),
                }
            )
            lastBHadron = BHadron[BHadron.hasBdaughter == 0]

            # 1) gluon splitting
            n_lastB = ak.num(lastBHadron)
            gsp_nom, gsp_up, gsp_down = get_gluonsplitting_weight_from_lastB(lastBHadron, pruned_ev.SelJet.pt)
            weights.add("gluonSplitting", gsp_nom, gsp_up, gsp_down)

            # 2) b fragmentation:
            genJetPt = ak.values_astype(
                ak.sum(pruned_ev.GenJet.pt * ak.all(pruned_ev.GenJet.metric_table(pruned_ev.SelJet) < 0.5, axis=2), axis=-1),
                float,
            )

            # Fraction of jet pT carried by the heaviest B hadron
            max_mass = ak.max(BHadron.mass, axis=-1)
            xB = ak.values_astype(
                (ak.num(BHadron) > 0) * ak.sum(BHadron.pT * (BHadron.mass == max_mass) / genJetPt, axis=-1),
                float,
            )

            bfrag_nom, bfrag_up, bfrag_down = get_bfragmentation_weight(xB, genJetPt)
            weights.add("bfragmentation", bfrag_nom, bfrag_up, bfrag_down)

            # 3) B-hadron decay (semi-leptonic BR) uncertainty
            bHadronId = ak.values_astype(
                -1 * (ak.num(lastBHadron) != 1)
                + (ak.num(lastBHadron) == 1) * ak.sum(lastBHadron.pdgID, axis=-1),
                float,
            )

            bdecay_nom, bdecay_up, bdecay_down = get_decay_weight(bHadronId)
            weights.add("bdecay", bdecay_nom, bdecay_up, bdecay_down)

            # 4) c fragmentation
            cfrag_nom, cfrag_up, cfrag_down = get_cfragmentation_weight(pruned_ev.GenPart, pruned_ev.SelJet)
            weights.add("cfragmentation", cfrag_nom, cfrag_up, cfrag_down)

            # 5) c->D (semi-muonic) fragmentation/decay uncertainty (no ancestry check)
            cd_nom, cd_up, cd_down = get_cdfragmentation_simple_weight(pruned_ev.GenPart, pruned_ev.SelJet)
            weights.add("cdfragmentation_simple", cd_nom, cd_up, cd_down)

            # 6) V0 (K0s / Lambda) systematic
            v0_nom, v0_up, v0_down = get_v0_weight(pruned_ev.GenPart, pruned_ev.SelJet)
            weights.add("v0", v0_nom, v0_up, v0_down)

        if isRealData:
            if self._year == "2022":
                run_num = "355374_362760"
            elif self._year == "2023":
                run_num = "366727_370790"
            elif self._year == "2024":
                run_num = "378985_386951"
            else:
                raise ValueError(f"Unsupported year {self._year} for prescales")

            # Decide one trigger per event based on SelJet.pt bin
            jet_pt = pruned_ev.SelJet.pt

            # build exclusive masks per trigger bin
            psweight = ak.zeros_like(jet_pt, dtype=np.float32)

            for _, cfg in pt_matrix.items():
                trig = cfg["trigger"]
                psfile = f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{trig}_run{run_num}.json"
                if not os.path.isfile(psfile):
                    raise NotImplementedError(
                        f"Prescale weights not available for {trig} in {self._year}. "
                        "Please run `scripts/dump_prescale.py`."
                    )

                in_pt_bin = (jet_pt >= cfg["ptmin"]) & (jet_pt < cfg["ptmax"])

                # require the trigger to have fired in the event (on pruned_ev!)
                fired = HLT_helper(pruned_ev, [trig])

                use = in_pt_bin & fired

                pseval = correctionlib.CorrectionSet.from_file(psfile)
                thisps = pseval["prescaleWeight"].evaluate(
                    pruned_ev.run,
                    f"HLT_{trig}",
                    ak.values_astype(pruned_ev.luminosityBlock, np.float32),
                )

                # add contribution only where this trigger is the chosen one
                psweight = ak.where(use, thisps, psweight)

            weights.add("psweight", ak.values_astype(psweight, float))
        if isRealData:
            frac0 = ak.mean(psweight == 0)
            print("psweight: min/max =", ak.min(psweight), ak.max(psweight), " frac==0 =", frac0)
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
