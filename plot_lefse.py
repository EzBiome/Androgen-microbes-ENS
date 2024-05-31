import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("demo_lefse_results.csv")

for u in df["Group"].unique():
    temp=df.loc[df["Group"]==u]
    feat=list(temp["Feature"])
    lda=list(temp["LDA Effect Size"])
    
    plt.barh(feat,lda,label=u)
    plt.legend()
    plt.title("Differentially Abundant Features (LEfSe)")
    plt.xlabel("LDA Score (log10)")
