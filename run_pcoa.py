
import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # for 3D plot
import sys, os
from glob import glob

color_dict = dict({'Group 1':'gray','Group 2':'orange'})

def pco_by_eigen(dist, n_components=2, return_eigen=False, return_variance_explained=False):

    # get some distance matrix
    A = dist
    # square it
    A = A**2

    # centering matrix
    n = A.shape[0]
    J_c = 1./n*(np.eye(n) - 1 + (n-1)*np.eye(n))

    # perform double centering
    B = -0.5*(J_c.dot(A)).dot(J_c)

    # find eigenvalues and eigenvectors
    eigen_val, eigen_vec = eigh(B) #the order is ascending, so need to flip
    eigen_val = np.flip(eigen_val)
    eigen_vec = np.flip(eigen_vec, axis=1)
    relative_eigen_val = 100*eigen_val/np.sum(eigen_val)

    # select dimensions
    eigen_val = eigen_val[:n_components]
    eigen_vec = eigen_vec[:,:n_components]
    # sign of eigenvector is usually randomly calculated, but it has huge effect on pcoa ordination.
    # so let's reverse the sign manually, not to affect previous contour graph
    eigen_vec[:,1] = -eigen_vec[:,1]
    relative_eigen_val = relative_eigen_val[:n_components]
    pco = np.sqrt(eigen_val)*eigen_vec
    if return_eigen:
        return (pco, (eigen_val, eigen_vec))
    elif return_variance_explained:
        return (pco, relative_eigen_val)
    else:
        return pco


def plot_pcoa_2d(df, treatment, percents):
    '''
    2D PCOA plotting. Distances are the same, the coloring is based on the treatment
    '''
    fig, ax = plt.subplots(figsize=(10,10), dpi=200)

    sns.scatterplot(data = df, 
                    x='1st PC', 
                    y='2nd PC',
                    hue=treatment,
                    palette=color_dict,
                    s=200
                   )
    
    ax.set_title(f"PCOA, Bray Curtis distance: {treatment}")
    xtit = f'1st PC: {percents[0]:.2f}%'
    ytit = f'2nd PC: {percents[1]:.2f}%'
    ax.set_xlabel(xtit)
    ax.set_ylabel(ytit)
    
    plt.savefig("Taxonomic_PCoA.png", bbox_inches='tight')
    with open("taxonomic_variance.txt","w") as var:
        for x in percents:
            var.write(f"{x}\n")
def main():

    sns.set_style('white')
    sns.set_palette('flare')

    metadata_file = "demo_metadata.csv"
    distance_file = "demo_distance_matrix.csv"

    dist_df = pd.read_csv(distance_file, header=0, index_col=0)
    meta_df = pd.read_csv(metadata_file, header=0,sep="\t", index_col=0)
    display(meta_df.head())
    # remove metadata for samples removed earlier
    meta_df = meta_df[meta_df.index.isin(list(dist_df.index))] 
    # make sure the metadata and the distance data have the same sorting
    meta_df = meta_df.sort_index(axis=0)
    dist_df = dist_df.sort_index(axis=0)
    dist_df = dist_df.sort_index(axis=1)
    dist_df = dist_df.apply(pd.to_numeric)

    treatments = list(meta_df.columns)

    coords, rel_var = pco_by_eigen(dist_df,n_components = 5, return_variance_explained = True )
    meta_df[['1st PC', '2nd PC', '3rd PC','4th PC','5th PC']] = coords 

    #os.makedirs('PCOA',exist_ok=True)
    meta_df.to_csv('Taxonomic_PCoA_coords.csv')

    for treatment in treatments:
        print(treatment)
        # remove samples not in either treatment group
        df_non_na = meta_df.dropna(axis=0, subset=[treatment])
        plot_pcoa_2d(df_non_na, treatment, rel_var)

if __name__ == "__main__":
    main() 
