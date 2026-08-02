## Most Influential Features on Sweetness Identified Through Regression and Classification Analyses

In the regression analysis performed with the XGBoost model to explain the variation 
in sweetness intensities, feature importance and SHAP analyses highlighted the most 
important molecular descriptors. The descriptors detailed below reflect the chemical 
and physical properties associated with sweetness intensity.

### AATSC0p (averaged and centered Moreau–Broto autocorrelation of lag 0 weighted by polarizability)
Autocorrelation molecular descriptors are used to measure the distribution of specific 
properties within a molecule according to topological distances (lags). These descriptors 
include the weighted product of properties between pairs of atoms separated by a given 
distance (lag k) and determine the distribution of properties among atoms within the 
molecule. The general Moreau–Broto autocorrelation formula is given below 
(Puzyn et al., 2010):

$$
ATS_k = \frac{1}{2N_k} \sum_{i=1}^{A} \sum_{j=1}^{A} w_i \cdot w_j \cdot \delta(d_{ij}, k)
$$

Where:  
* $A$ is the total number of atoms in the molecule,
* $w_i$ and $w_j$ represent atomic properties (e.g., polarizability),
* $d_{ij}$ is the topological distance (number of bonds) between atoms,
* $\delta(d_{ij}, k)$ is the Kronecker delta function (1 if $d_{ij}=k$, otherwise 0),
* $N_k$ is the number of atom pairs at distance $k$.

This formula gives the autocorrelation of atom pairs in terms of a specific property 
based on their topological distance.

AATSC0p is a Moreau–Broto autocorrelation descriptor with lag 0, weighted by 
polarizability. Lag 0 includes the squared polarizability values of all atoms in the 
molecule, and the average of these values is taken. When $k = 0$, the Kronecker 
delta only includes each atom's own squared polarizability value.

$$
ATS_0 = \sum_{i=1}^{A} w_i^2
$$

This expression gives the sum of the squared polarizability values of all atoms and 
reflects the general distribution of polarizability within the molecule. To reduce 
correlation among descriptors and to make them more suitable for statistical analyses, 
centered (mean-subtracted) polarizability values are used.

The formula for the AATSC descriptor is as follows:

$$
AATSC_k = \frac{ATS_k}{\Delta_k}
$$

Where:  
* $AATSC_k$ is the averaged autocorrelation at distance $k$,  
* $ATSC_k$ denotes the total autocorrelation value at distance $k$,  
* $\Delta_k$ denotes the number of node pairs at distance $k$.

This formula yields the average autocorrelation value by normalizing $ATSC_k$ with 
$\Delta_k$, allowing the distribution of atomic properties by topological distance 
within the molecule to be analysed in a mean-centred manner (Mordred, 2026).

---

### AATS1p (averaged Moreau–Broto autocorrelation of lag 1 weighted by polarizability)

This molecular descriptor represents the mean of the polarizability‑weighted autocorrelation 
for atom pairs that are one bond apart (topological distance 1) within the molecule (Mordred, 2026).

---

### ETA_dPsi_A

A descriptor that measures a molecule's propensity to form hydrogen bonds  
(Gooch et al., 2017; Mordred, 2026).

---

### ETA_dEpsilon_D

ETA_dEpsilon_D is an extended topochemical index that quantifies the contribution of hydrogen‑bond‑donor atoms. 
A negative coefficient indicates that a higher number of hydrogen‑bond donors in the compound (typically atoms 
such as N, O, or F with high electronegativity, and thus higher polarity) exerts a stronger effect  
(De & Roy, 2018; Mordred, 2026).

---

### AATS0p (averaged Moreau–Broto autocorrelation of lag 0 weighted by polarizability)

AATS0p gives the average of the squared polarizability values of all atoms in a molecule. For lag 0, instead of a 
topological distance between two atoms, each atom's own squared polarizability value is used and averaged (Mordred, 2026).

---

### MPC6 (6‑ordered path count)

This descriptor expresses the number of bond paths of a specified length within the 2‑D structure of the molecule. 
With the parameter order = 6, it counts all paths containing six bonds. Focusing solely on direct path counts, 
it ignores π‑electrons and the total path count from 1 up to the order value, and no logarithmic scale is 
applied (Mordred, 2026).

---

### MWC10 (walk count – leg-10)

MWC10 is the number of molecular walks of order 10 and is related to the complexity of the molecular graph in terms of branching and size. 
In other words, larger and more complex molecules possess higher MWC10 values 
(Dieguez-Santana et al., 2016; Mordred, 2026).

---

### piPC7 (7‑ordered π-path count – log scale)

A molecular descriptor giving the logarithmic count of π-electrons along bond paths of length 7 in the molecule (Mordred, 2026).

---

### ETA_dAlpha_B

The ETA_dAlpha_B descriptor measures the presence of hydrogen-bond acceptor atoms and/or the polar surface area within a molecule 
(Mordred, 2026; Seth et al., 2020).

---

### piPC5 (5‑ordered π-path count – log scale)

piPC5 is a molecular descriptor that provides the logarithmic count of π-electrons along bond paths of length 5 in the molecule 
(Mordred, 2026).

---

### GATS2p (Geary coefficient of lag 2 weighted by polarizability)

GATS2p is a molecular descriptor representing the Geary coefficient at a topological distance (lag) of 2, weighted by polarizability. 
Geary's coefficient is an autocorrelation function originally used in ecological studies to measure the spatial distribution of 
environmental properties and is applied to molecular structures in a manner analogous to the Moreau–Broto function. 
The Moran and Geary functions provide true autocorrelation by taking into account mean and standard deviation values. 
The Geary coefficient $c_k$ is defined as follows:

$$
c_k = \frac{\displaystyle\frac{1}{2V_k}\sum_{i=1}^{A}\sum_{j=1}^{A}(w_i - w_j)^2\,\delta(d_{ij},k)}{\displaystyle\frac{1}{A-1}\sum_{i=1}^{A}(w_i - \bar{w})^2}
$$

Where:  
* $w_i$ represents any atomic property on the molecule  
* $\bar{w}$ is the mean value of that property across the molecule  
* $A$ is the number of atoms  
* $k$ is the specified lag value (topological distance)  
* $d_{ij}$ is the topological distance between the *i*-th and *j*-th atoms  
* $\delta(d_{ij}, k)$ is the Kronecker delta function (1 if $d_{ij} = k$, else 0)

The Geary coefficient is a distance-type function that ranges from zero to infinity. A strong autocorrelation yields low index values. 
Positive autocorrelation gives values between 0 and 1, while negative autocorrelation gives values greater than 1. When no correlation 
is present, the reference value is taken as $c_k = 1$ (Puzyn et al., 2010).

---

### MWC09 (walk count – leg-9)

MWC09 is the number of molecular walks of order 9 and is related to the complexity of the molecular graph in terms of branching and size 
(Mordred, 2026).

---

### MPC4 (4-ordered path count)

MPC4 is a PathCount descriptor that denotes the number of bond paths of length 4 in the two-dimensional structure of the molecule. 
This descriptor counts only direct paths and does not consider π-electrons, the total number of paths, or any logarithmic scale 
(Mordred, 2026).

---

### SssNH (sum of ssNH)

SssNH is a molecular descriptor that gives the sum of E-State values for the ssNH atom type (an sp³-hybridised nitrogen–hydrogen (NH) group 
with two sigma bonds) (Mordred, 2026).

---

### AATSC1are (averaged and centered Moreau–Broto autocorrelation of lag 1 weighted by Allred–Rochow EN)

AATSC1are is an averaged, centered Moreau–Broto autocorrelation index at lag 1, weighted by Allred–Rochow electronegativity within the 
topological structure of a molecule. Autocorrelation indices describe how a given property is distributed throughout the molecular 
structure (Ignacz & Szekely, 2022; Mordred, 2026). Allred–Rochow electronegativity represents the attractive force between an electron 
and the nucleus, accounting for shielding and covalent radius (Accorinti & Labarca, 2020).

---

### SpDiam_A (SpDiam of adjacency matrix)

SpDiam_A denotes the spectral diameter value obtained from the topological distance matrix (Mordred, 2026).

---

### AATS3p (averaged Moreau–Broto autocorrelation of lag 3 weighted by polarizability)

AATS3p is a molecular descriptor that expresses the average Moreau–Broto autocorrelation coefficient at a topological distance of three bonds (lag 3), 
weighted by polarizability (Mordred, 2026).

---

### NssNH (number of ssNH)

NssNH is an electron-state (E-State) descriptor that counts the number of ssNH groups (nitrogen–hydrogen (NH) groups that are sp³-hybridised and form 
two sigma bonds) present in a molecule (Mordred, 2026).

---

### FilterItLogS (Filter-it™ LogS)

FilterItLogS denotes the logarithmic aqueous solubility (LogS) of a molecule as calculated by the Filter-it™ software (Mordred, 2026).

---

### BCUTi-1h (first highest eigenvalue of Burden matrix weighted by ionisation potential)

BCUTi-1h is a BCUT descriptor that gives the highest first eigenvalue (n = 0) of the Burden matrix weighted by ionisation potential (Mordred, 2026).

BCUT metrics extend Burden's work, which represents a molecule's hydrogen-stripped connection table as a symmetric $N \times N$ matrix whose diagonal 
entries contain atomic numbers and off-diagonal elements contain bonding information (Burden, 1989).

Pearlman and colleagues extended this approach by using the highest and lowest eigenvalues of four classes of BCUT matrices to expand into a multi-dimensional space; the diagonals of these matrices contain atomic properties important for ligand–receptor interactions: atomic charges, polarizabilities, hydrogen-bond acceptor and donor abilities 
(Pearlman & Smith, 1998; Pirard & Pickett, 2000).

---

### MATS1pe (Moran coefficient of lag 1 weighted by Pauling EN)

MATS1pe is a molecular descriptor expressing the Moran autocorrelation coefficient for atom pairs separated by one bond (lag 1), weighted by Pauling electronegativity. The Moran coefficient measures how atomic properties are distributed within a molecule and is given by the following equation
(Mordred, 2026; Puzyn et al., 2010):

$$
I_k = \frac{\displaystyle\frac{1}{V_k}\sum_{i=1}^{A}\sum_{j=1}^{A}(w_i - \bar{w})(w_j - \bar{w})\,\delta(d_{ij},k)}{\displaystyle\frac{1}{A}\sum_{i=1}^{A}(w_i - \bar{w})^{2}}
$$

Where:  
* $V_k$ is the number of atom pairs at lag $k$  
* $w_i$, $w_j$ are the atomic properties under consideration  
* $\bar{w}$ is the mean value of that property over all atoms  
* $d_{ij}$ is the topological distance between atoms $i$ and $j$  
* $\delta(d_{ij}, k)$ is the Kronecker delta (1 if $d_{ij}=k$, otherwise 0)  
* $A$ is the total number of atoms in the molecule

The Moran coefficient generally takes a value in the range $[-1, +1]$. In this equation, $w_i$ represents the Pauling electronegativity of atom $i$, and $\bar{w}$ denotes the mean electronegativity across all atoms in the molecule. Positive autocorrelation corresponds to positive values of the coefficient, 
whereas negative autocorrelation yields negative values (Puzyn et al., 2010).

---

### AATS0i (averaged Moreau–Broto autocorrelation of lag 0 weighted by ionization potential)

AATS0i is a 2-D descriptor that expresses the average Moreau–Broto autocorrelation coefficient at lag 0, weighted by ionization potential. For lag 0, 
each atom's own property value is autocorrelated, and these values are averaged according to the AATS formula (Mordred, 2026).

---

### SlogP_VSA10 (MOE logP VSA Descriptor 10, 0.40 ≤ x < 0.50)

SlogP_VSA10 is a MOE (Molecular Operating Environment) descriptor based on Wildman–Crippen logP values and surface-area contributions. The logP range 
0.40–0.50 represents the summed surface areas of atoms whose logP contributions fall within that interval. LogP is a parameter describing a molecule's lipophilicity (fat solubility). This descriptor does not require kekulisation of the molecule and ignores hydrogens that are not explicitly bonded 
(Mordred, 2026).

---

### AXp-0dv (0-ordered averaged Chi path weighted by valence electrons)

AXp-0dv is a zeroth-order Chi descriptor in which each atom in the molecule is weighted by its valence electrons. It is calculated to evaluate the molecule's topological properties and is used without normalisation (Mordred, 2026).

---

### ATSC2c (centered Moreau–Broto autocorrelation of lag 2 weighted by Gasteiger charge)

ATSC2c is a centered Moreau–Broto autocorrelation descriptor based on the topological distance (lag 2) between atom pairs within the molecule, weighted by Gasteiger charge. It is used to analyse the distribution of atomic charges at a specified distance in the molecular structure (Mordred, 2026).

---

In the classification analysis performed with the XGBoost model to distinguish between sweet and non-sweet compounds, feature importance analysis identified the most important molecular descriptors. The descriptors listed below reflect the most important molecular features in discriminating between sweet and non-sweet compounds.

---

### AATSC4c (averaged and centered Moreau–Broto autocorrelation of lag 4 weighted by Gasteiger charge)

AATSC4c is a molecular descriptor that expresses the averaged, mean-centered Moreau–Broto autocorrelation coefficient calculated at a topological distance of 
four bonds (lag 4), weighted by Gasteiger charge (Mordred, 2026).

---

### ATSC0c (centered Moreau–Broto autocorrelation of lag 0 weighted by Gasteiger charge)

ATSC0c provides the mean-centered Moreau–Broto autocorrelation value for a molecule at lag 0 (i.e. on the same atom), weighted by Gasteiger charge 
(Mordred, 2026).

---

### AATS0p (averaged Moreau–Broto autocorrelation of lag 0 weighted by polarizability)

AATS0p gives the average of the squared polarizability values of all atoms in a molecule. For lag 0, instead of a topological distance between two atoms, each atom's own squared polarizability value is used and averaged (Mordred, 2026).

---

### ATS5are (Moreau–Broto autocorrelation of lag 5 weighted by Allred–Rochow EN)

ATS5are is a molecular descriptor representing the Moreau–Broto autocorrelation coefficient calculated at a topological distance of five bonds (lag 5) and weighted by Allred–Rochow electronegativity (Mordred, 2026).

---

### AATSC0c (averaged and centered Moreau–Broto autocorrelation of lag 0 weighted by Gasteiger charge)

AATSC0c denotes the averaged, mean-centered Moreau–Broto autocorrelation coefficient at lag 0 (on the same atom) weighted by Gasteiger charge (Mordred, 2026).

---

### ATSC0are (centered Moreau–Broto autocorrelation of lag 0 weighted by Allred–Rochow EN)

ATSC0are is a molecular descriptor that gives the mean-centered Moreau–Broto autocorrelation coefficient at lag 0, weighted by Allred–Rochow electronegativity 
(Mordred, 2026).

---

### BalabanJ (Balaban's J index)

Balaban's J index is defined as a topological index that summarizes connectivity and distance information in a molecular graph (Mordred, 2026). It is calculated from the connection (adjacency) matrix using row sums, and its formula is:

$$
J = \frac{q}{\mu + 1}\left(\sum_{\text{edges } ij} S_i S_j\right)^{-\tfrac{1}{2}}
$$

Where:  
* $q$ is the number of edges in the molecular graph  
* $\mu = q - n + 1$ is the cyclomatic number of the graph  
* $n$ is the number of atoms (vertices)  
* $S_i$ is the row or column sum of the topological distance matrix for atom $i$  
(Balaban, 1981, 1983; CODESSA-PRO, 2024)

Calculations based on the connection matrix provide information about theoretical values of atoms in the molecule and help distinguish primary, secondary, tertiary, and quaternary carbon atoms (Randić & Pompe, 2001).

---

### TopoPSA (topological polar surface area)

TopoPSA is a descriptor that calculates a molecule's topological polar surface area (TPSA) and is related to the polarity of chemicals (Mordred, 2026; 
Takaku et al., 2015). The descriptor allows the TPSA calculation to include only nitrogen (N) and oxygen (O) atoms, or optionally to include nitrogen, 
oxygen, sulfur (S), and phosphorus (P) atoms (Mordred, 2026).

---

### AATSC4v (averaged and centered Moreau–Broto autocorrelation of lag 4 weighted by vdW volume)

AATSC4v is a molecular descriptor that expresses the averaged, mean-centered Moreau–Broto autocorrelation coefficient calculated at a topological distance of 
four bonds (lag 4), weighted by the van der Waals volume of the atoms (Mordred, 2026).

---

### naRing (RingCount – aromatic ring count)

naRing is a molecular descriptor that indicates the number of aromatic rings present in a molecule (Mordred, 2026).

---

### ATS6m (Moreau–Broto autocorrelation of lag 6 weighted by mass)

ATS6m is a molecular descriptor that calculates the Moreau–Broto autocorrelation at a topological distance of six bonds (lag 6), weighted by the atomic masses of the atoms involved (Mordred, 2026).

---

### Mv (mean of constitutional weighted by vdW volume)

Mv is a constitutional descriptor that gives the average of the van der Waals volumes of the atoms in a molecule, weighted by those volumes (Mordred, 2026).

---

### MATS4dv (Moran coefficient of lag 4 weighted by valence electrons)

MATS4dv is a molecular descriptor that calculates the Moran autocorrelation coefficient at a topological distance of four bonds (lag 4), weighted by valence electrons (Mordred, 2026).

---

### AATSC1v (averaged and centered Moreau–Broto autocorrelation of lag 1 weighted by vdW volume)

AATSC1v is a molecular descriptor that expresses the averaged, mean-centered Moreau–Broto autocorrelation coefficient at a topological distance of one bond 
(lag 1), weighted by the van der Waals volume of the atoms (Mordred, 2026).

---

### AATS2dv (averaged Moreau–Broto autocorrelation of lag 2 weighted by valence electrons)

AATS2dv is a molecular descriptor that calculates the average Moreau–Broto autocorrelation coefficient at a topological distance of two bonds (lag 2), weighted 
by valence electrons (Mordred, 2026).

---

## References

Accorinti, H., & Labarca, M. (2020). Commentary on the Models of Electronegativity. *Journal of Chemical Education*, *97*(10), 3474–3477. https://doi.org/10.1021/acs.jchemed.0c00512

Balaban, A. T. (1981). No Title. *Chemical Physics Letters*, *89*, 399.

Balaban, A. T. (1983). No. *Pure and Applied Chemistry*, *55*, 199.

Burden, F. R. (1989). Molecular Identification Number for Substructure Searches. *Journal of Chemical Information and Computer Sciences*, *29*, 225–227.

CODESSA-PRO. (2024). *Balaban's J index*. https://www.codessa-pro.com/descriptors/topo/balaban.htm

De, P., & Roy, K. (2018). Greener chemicals for the future: QSAR modelling of the PBT index using ETA descriptors. *SAR and QSAR in Environmental Research*, *29*(4), 319–337. https://doi.org/10.1080/1062936X.2018.1436086

Dieguez-Santana, K., Pham-The, H., Villegas-Aguilar, P. J., Le-Thi-Thu, H., Castillo-Garit, J. A., & Casañola-Martin, G. M. (2016). Prediction of acute toxicity of phenol derivatives using multiple linear regression approach for Tetrahymena pyriformis contaminant identification in a median-size database. *Chemosphere*, *165*, 434–441. https://doi.org/10.1016/j.chemosphere.2016.09.041

Gooch, A., Sizochenko, N., Rasulev, B., Gorb, L., & Leszczynski, J. (2017). In vivo toxicity of nitroaromatics: A comprehensive quantitative structure–activity relationship study. *Environmental Toxicology and Chemistry*, *36*(8), 2227–2233. https://doi.org/10.1002/etc.3761

Ignacz, G., & Szekely, G. (2022). Deep learning meets quantitative structure–activity relationship (QSAR) for leveraging structure-based prediction of solute rejection in organic solvent nanofiltration. In *Journal of Membrane Science* (Vol. 646). https://doi.org/10.1016/j.memsci.2022.120268

Mordred. (2026). *Descriptor list (Mordred molecular descriptor calculator)*. https://mordred-descriptor.github.io/documentation/master/descriptors.html

Moriwaki, H., Tian, Y. S., Kawashita, N., & Takagi, T. (2018). Mordred: A molecular descriptor calculator. *Journal of Cheminformatics*, *10*(1), 1–14. https://doi.org/10.1186/s13321-018-0258-y

Pearlman, R. S., & Smith, K. M. (1998). Novel Software Tools for Chemical Diversity. In Y. C. Kubinyi, H., Folkers, G., & Martin (Ed.), *Perspective in Drug Discovery and Design* (p. 339). Kluwer/Escom.

Pirard, B., & Pickett, S. D. (2000). Classification of Kinase Inhibitors Using BCUT Descriptors. *Journal of Chemical Information and Computer Sciences*, *40*(6), 1431–1440. https://doi.org/10.1021/ci000386x

Puzyn, T., Leszczyński, J., & Cronin, M. (2010). Recent Advances in QSAR Studies: Methods and Applications. In *... Advances in Computational Chemistry ...*. http://link.springer.com/content/pdf/10.1007/978-1-4020-9783-6.pdf

Randić, M., & Pompe, M. (2001). The Variable Molecular Descriptors Based on Distance Related Matrices. *Journal of Chemical Information and Computer Sciences*, *41*(3), 575–581. https://doi.org/10.1021/ci0001029

Seth, A., Ojha, P. K., & Roy, K. (2020). QSAR modeling with ETA indices for cytotoxicity and enzymatic activity of diverse chemicals. *Journal of Hazardous Materials*, *394*, 122498. https://doi.org/10.1016/j.jhazmat.2020.122498

Takaku, T., Nagahori, H., Sogame, Y., & Takagi, T. (2015). Quantitative structure-activity relationship model for the fetal-maternal blood concentration ratio of chemicals in humans. *Biological and Pharmaceutical Bulletin*, *38*(6), 930–934. https://doi.org/10.1248/bpb.b14-00883
