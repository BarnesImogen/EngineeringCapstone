import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from signature_definitions import (
    RISK_CLASSES,
    ONCOTYPE_GROUPS, ONCOTYPE_SINGLE_GENES, ONCOTYPE_REFERENCE_GENES, ONCOTYPE_GENE_SCALE, ONCOTYPE_REFERENCE_OFFSET,
    ONCOTYPE_RSU_OFFSET, ONCOTYPE_RSU_MULTIPLIER,
    PAM50_ALIASES, ROR_S_COEFFICIENTS,
    BCI_HI_GENES, BCI_MGI_GENES, BCI_MGI_WEIGHT, BCI_HI_WEIGHT, BCI_HI_CUBIC_WEIGHT,
    IRRS7_WEIGHTS, HU11_WEIGHTS,
)

# Fixed seed for drawing the ER-balanced PAM50 centring subset
PAM50_CENTRING_SEED = 42

# Oncotype DX ✅
# PAM50 ✅
# Breast Cancer Index (BCI) ✅
# IRRS7 ✅
# Hu11 ✅

def _require_genes(df, genes, signature):
    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise ValueError(f"{signature}: missing genes {missing}")
    print(f"{signature}: Found {len(genes)}/{len(genes)} genes")

def classify_by_cutoffs(score, cutoffs):
    """Low below the first cutoff, Intermediate up to (not including) the second, High from the second."""
    low, high = cutoffs
    return pd.Series(np.select([score < low, score < high], RISK_CLASSES[:2], RISK_CLASSES[2]), index=score.index)

def classify_by_tertiles(score):
    """Low / Intermediate / High by cohort tertiles of the score."""
    return pd.qcut(score, 3, labels=RISK_CLASSES).astype(str)

def infer_er_status(esr1):
    """
    ER status inferred from ESR1 expression, which is bimodal in TCGA-BRCA.

    Fits a two-component Gaussian mixture to log2 ESR1 and calls a patient ER-positive when
    the posterior of the higher-mean component is >= 0.5. Also reports the antimode (density
    minimum between the two component means) as a cross-check of the threshold.

    Returns (status Series of "Positive"/"Negative", mixture threshold, antimode).
    """
    x = esr1.to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
    positive = int(np.argmax(gmm.means_.ravel()))
    is_positive = gmm.predict_proba(x)[:, positive] >= 0.5

    low_mean, high_mean = np.sort(gmm.means_.ravel())
    grid = np.linspace(low_mean, high_mean, 2001)
    grid_positive = gmm.predict_proba(grid.reshape(-1, 1))[:, positive]
    threshold = float(grid[np.argmax(grid_positive >= 0.5)])
    antimode = float(grid[np.argmin(gaussian_kde(esr1)(grid))])

    status = pd.Series(np.where(is_positive, "Positive", "Negative"), index=esr1.index)
    return status, threshold, antimode

def calculate_oncotype_dx_score(df, scaling="minmax"):
    """
    Oncotype DX 21-gene Recurrence Score (Paik et al., 2004), following genefu's oncotypedx.

    1. Reference normalisation (kept from the published assay; genefu drops it): subtract the
       patient's mean of ACTB, GAPDH, GUSB, RPLP0, TFRC. The data is log2, so this is a
       per-patient subtraction.
    2. Rescale each gene to 0-15 across the cohort. scaling="minmax" uses the cohort min and
       max, as genefu does. scaling="p1_99" uses the 1st and 99th percentiles and bounds the
       result to [0, 15], which is less sensitive to single outliers.
       scaling="reference_plus_10" skips the cohort step and uses the reference-normalised
       log2 value + 10.
    3. Group scores:
         GRB7 group          = 0.9*GRB7 + 0.1*ERBB2, floored at 8
         ER group            = (0.8*ESR1 + 1.2*PGR + BCL2 + SCUBE2) / 4
         Proliferation group = mean(BIRC5, MKI67, MYBL2, CCNB1, AURKA), floored at 6.5
         Invasion group      = mean(CTSL2, MMP11)
    4. RSU = 0.47*GRB7 group - 0.34*ER group + 1.04*Proliferation group + 0.10*Invasion group
             + 0.05*CD68 - 0.08*GSTM1 - 0.07*BAG1
    5. RS = 20 * (RSU - 6.7), bounded to [0, 100] (genefu applies the published piecewise rule
       and then bounds the result, so RSU < 6.7 gives 0 rather than a negative RS).

    Returns (RSU, RS). RSU is the continuous score used everywhere downstream (classes,
    heatmap, survival); RS is kept as a separate column for reference only.

    Risk classes: cohort tertiles of the unbounded RSU, not the published RS cutoffs (Low < 18,
    Intermediate 18 to < 31, High >= 31). Those cutoffs assume the assay's own qPCR scale. On
    this RNA-seq data they put 67% (min-max), 54% (1st-99th percentile) or 45% (reference + 10)
    of ER+ patients in High, against roughly 15-27% in published ER+ cohorts. RS is bounded at
    0 and 100, which ties about a third of the cohort at RS = 100, so tertiles are taken on
    RSU, which keeps the full ranking.
    """
    genes = [g for gene_weights, _, _ in ONCOTYPE_GROUPS.values() for g in gene_weights] + list(ONCOTYPE_SINGLE_GENES)
    _require_genes(df, genes, "Oncotype DX")
    _require_genes(df, ONCOTYPE_REFERENCE_GENES, "Oncotype DX reference")

    normalised = df[genes].sub(df[ONCOTYPE_REFERENCE_GENES].mean(axis=1), axis=0)

    if scaling == "reference_plus_10":
        scaled = normalised + ONCOTYPE_REFERENCE_OFFSET
    elif scaling in ("minmax", "p1_99"):
        if scaling == "minmax":
            lower, upper = normalised.min(), normalised.max()
        else:
            lower, upper = normalised.quantile(0.01), normalised.quantile(0.99)
        scaled = ((normalised - lower) / (upper - lower)).clip(0, 1) * ONCOTYPE_GENE_SCALE
    else:
        raise ValueError(f"Unknown Oncotype DX scaling: {scaling}")

    rsu = pd.Series(0.0, index=df.index)
    for gene_weights, weight, floor in ONCOTYPE_GROUPS.values():
        group_score = sum(scaled[gene] * w for gene, w in gene_weights.items())
        if floor is not None:
            group_score = group_score.clip(lower=floor)
        rsu += weight * group_score
    rsu += sum(scaled[gene] * w for gene, w in ONCOTYPE_SINGLE_GENES.items())

    rs = (ONCOTYPE_RSU_MULTIPLIER * (rsu - ONCOTYPE_RSU_OFFSET)).clip(0, 100)
    return rsu, rs

def fetch_pam50_centroids():
    """
    Uses the rpy2 bridge to extract the official Parker 2009
    centroid matrix from the Bioconductor genefu package.
    """
    r_script = """
    function() {
        # Load the pam50 object from genefu
        data(pam50, package="genefu")

        # Convert it to an R data.frame so pandas translates the gene names perfectly
        centroid_df <- as.data.frame(pam50$centroids)
        return(centroid_df)
    }
    """
    get_centroids = robjects.r(r_script)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        # Because we are inside the localconverter block, get_centroids()
        # returns a fully formatted pandas DataFrame automatically!
        centroid_df = get_centroids()

    # genefu uses 'Her2', but our Python math formula explicitly looks for 'HER2'
    # We rename the columns here to ensure the dictionary lookup doesn't crash
    centroid_df.columns = ['Basal', 'HER2', 'LumA', 'LumB', 'Normal']

    return centroid_df

def _row_standardised_ranks(values):
    """Rank each row, then centre and scale it, so a dot product / n is a Spearman correlation."""
    ranks = values.rank(axis=1)
    return ranks.sub(ranks.mean(axis=1), axis=0).div(ranks.std(axis=1, ddof=0), axis=0)

def calculate_pam50_score(df, centroid_df, er_status, seed=PAM50_CENTRING_SEED):
    """
    PAM50 nearest-centroid subtype and ROR-S (Parker et al., 2009; genefu rorS).

    df: patient data (rows=patients, cols=genes)
    centroid_df: Parker 2009 centroids (rows=50 genes, cols=5 subtypes)
    er_status: inferred ER status per patient ("Positive"/"Negative")

    1. ER-balanced median centring: gene medians come from a subset with equal numbers of ER+
       and ER- patients (all of the smaller group plus a seeded random draw of the same size
       from the larger), and are subtracted from every patient. Whole-cohort centring would
       shift the centre towards ER+ tumours, which dominate TCGA-BRCA.
    2. Spearman correlation of each patient with each centroid; the subtype is the highest.
    3. ROR-S = 0.05*Basal + 0.12*HER2 - 0.34*LumA + 0.23*LumB (no Normal term).
    4. Rescaled to 0-100 as genefu does: (ROR-S - q2.5) / (q97.5 - q2.5) * 100, bounded to
       [0, 100], with the cohort's 2.5th and 97.5th percentiles.

    Risk classes: Low < 29, Intermediate 29 to < 53, High >= 53.
    """
    # The genefu centroid matrix uses older gene symbols (CDCA1, KNTC2); map them to the
    # RNA-seq column names before matching.
    centroid_df = centroid_df.rename(index=PAM50_ALIASES)
    genes = centroid_df.index.tolist()
    available_genes = [g for g in genes if g in df.columns]

    print(f"PAM50: Found {len(available_genes)}/{len(genes)} genes")
    if len(available_genes) < len(genes):
        print(f"Missing: {sorted(set(genes) - set(available_genes))}")

    er_status = er_status.reindex(df.index)
    positives = er_status.index[er_status == "Positive"]
    negatives = er_status.index[er_status == "Negative"]
    n_each = min(len(positives), len(negatives))
    rng = np.random.default_rng(seed)
    subset = np.concatenate([rng.choice(positives, n_each, replace=False), rng.choice(negatives, n_each, replace=False)])
    print(f"PAM50: centring on {n_each} ER+ and {n_each} ER- patients (seed {seed})")

    centered_df = df[available_genes] - df.loc[subset, available_genes].median(axis=0)

    patient_ranks = _row_standardised_ranks(centered_df)
    centroid_ranks = _row_standardised_ranks(centroid_df.loc[available_genes].T)
    correlations = patient_ranks @ centroid_ranks.T / len(available_genes)

    subtypes = correlations.idxmax(axis=1)

    ror_s_raw = sum(correlations[subtype] * weight for subtype, weight in ROR_S_COEFFICIENTS.items())
    q_low, q_high = ror_s_raw.quantile(0.025), ror_s_raw.quantile(0.975)
    ror_s = ((ror_s_raw - q_low) / (q_high - q_low) * 100).clip(0, 100)

    return subtypes, ror_s

def calculate_bci_score(df, er_status, mgi_scaling="pc"):
    """
    Breast Cancer Index (Ma et al., 2008; Habel et al., 2013, Breast Cancer Research).

    The exact Biotheranostics assay is proprietary; this follows the published model.

    MGI: BUB1B, CENPA, NEK2, RACGAP1 and RRM2 standardised to mean 0, SD 1, combined as the
         first principal component, oriented so higher means more proliferation.
         Ma et al. (2008) do not say whether the component is rescaled afterwards. The
         default, mgi_scaling="pc", keeps the raw component score (the literal reading; on
         ER+ patients it gives 50/25/25% Low/Intermediate/High against 50/23/27% in Habel
         et al. 2013). Alternatives for sensitivity analysis:
           mgi_scaling="unit" divides it by its SD (sqrt of the first eigenvalue) so MGI has
                              SD 1 like H/I;
           mgi_scaling="mean" uses the mean of the standardised genes instead of the PCA.
    H/I: HOXB13 - IL17RB on the log2 scale (IL17RB is the current symbol for IL17BR),
         standardised to mean 0, SD 1.
    BCI raw = 0.4431*MGI + 0.4972*HI - 0.09*HI^3, rescaled to 0-10 as
         0 if raw < -2.5, 2*raw + 5 if -2.5 <= raw <= 2.5, 10 if raw > 2.5.

    Deviation from the published assay: the original standardised MGI genes and H/I with fixed
    constants from its own reference set, which is not available. Here the means and SDs, and
    the principal component, are fitted on the ER-positive patients only (inferred ER status)
    and applied to all patients. BCI was developed on ER-positive patients, so this keeps the
    scale close to the original reference population.

    Risk classes: Low < 5, Intermediate 5 to < 6.4, High >= 6.4.
    """
    _require_genes(df, BCI_HI_GENES + BCI_MGI_GENES, "BCI")
    is_reference = (er_status.reindex(df.index) == "Positive").to_numpy()

    mgi_genes = df[BCI_MGI_GENES]
    reference = mgi_genes[is_reference]
    z = (mgi_genes - reference.mean()) / reference.std()

    eigenvalues, eigenvectors = np.linalg.eigh(np.corrcoef(reference.to_numpy().T))
    loading, first_eigenvalue = eigenvectors[:, -1], eigenvalues[-1]
    # Higher MGI = more proliferation: every proliferation gene should load positively
    if loading.sum() < 0:
        loading = -loading
    pc = z @ loading

    if mgi_scaling == "unit":
        mgi = pc / np.sqrt(first_eigenvalue)
    elif mgi_scaling == "pc":
        mgi = pc
    elif mgi_scaling == "mean":
        mgi = z.mean(axis=1)
    else:
        raise ValueError(f"Unknown MGI scaling: {mgi_scaling}")

    hi_ratio = df[BCI_HI_GENES[0]] - df[BCI_HI_GENES[1]]
    hi = (hi_ratio - hi_ratio[is_reference].mean()) / hi_ratio[is_reference].std()

    bci_raw = BCI_MGI_WEIGHT * mgi + BCI_HI_WEIGHT * hi + BCI_HI_CUBIC_WEIGHT * hi ** 3
    return (2 * bci_raw + 5).clip(0, 10)

def calculate_irrs7_score(df):
    """
    IRRS-7 Signature (MDPI/Frontiers, 2025)
    Focus: Insulin Resistance-Related Prognostic Score.

    Classes come from cohort tertiles. The original paper used a median split; tertiles are a
    deliberate deviation so every signature in the panel has Low / Intermediate / High classes.
    """
    weights = IRRS7_WEIGHTS

    available_genes = [g for g in weights.keys() if g in df.columns]

    score = df[available_genes].apply(
        lambda row: sum(row[g] * weights[g] for g in available_genes), axis=1
    )
    return score

def calculate_hu11_irg_score(df):
    """
    Hu-11 IRG Signature (Hu et al., 2024)
    Focus: Inflammation-Related Genes and immune microenvironment.

    Classes come from cohort tertiles. The original paper used a median split; tertiles are a
    deliberate deviation so every signature in the panel has Low / Intermediate / High classes.
    """
    weights = HU11_WEIGHTS

    available_genes = [g for g in weights.keys() if g in df.columns]

    score = df[available_genes].apply(
        lambda row: sum(row[g] * weights[g] for g in available_genes), axis=1
    )
    return score
