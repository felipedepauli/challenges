import numpy as np
import pandas as pd

def preprocessing(df, hypothesis, test=False):

    # TASK 01 - Replace all strange numbers to np.nan
    
    df.replace({'nan': np.nan, None: np.nan}, inplace=True)
    df.replace(r'^(nan|NA|none|null)$', np.nan, regex=True, inplace=True)
    df.isna().sum()
    
    # ----------------------------------------------------
    
    
    # TASK 02 - Split columns
    
    # GroupSize
    df['Group']     = df['PassengerId'].apply(lambda x: x.split("_")[0])
    df['GroupSize'] = df['Group'].map(df['Group'].value_counts())
    df.drop("Group", axis=1, inplace=True)
    
    # Cabin
    df[["Cabin_Deck", "Cabin_Num", "Cabin_Size"]] = df["Cabin"].str.split('/', expand=True)
    df.drop("Cabin", axis=1, inplace=True)
    
    df["Cabin_Num"].fillna(method='ffill', inplace=True)
    df["Cabin_Num"] = df["Cabin_Num"].astype(np.uint8)
    
    # Name
    df.drop("Name", axis=1, inplace=True)
    
    # ----------------------------------------------------
    
    # TASK 03 - Dummies
    
    df = pd.get_dummies(df, "HP", "_",      columns=["HomePlanet"],  dummy_na=True)
    df = pd.get_dummies(df, "Dest", "_",    columns=["Destination"], dummy_na=True)
    df = pd.get_dummies(df, "C_Deck", "_",  columns=["Cabin_Deck"],  dummy_na=True)
    df = pd.get_dummies(df, "C_Size", "_",  columns=["Cabin_Size"],  dummy_na=True)

    # ----------------------------------------------------
    
    # TASK 04 - Binaries
    
    df["CryoSleep"] = df["CryoSleep"].apply(lambda x: 1 if "True" else 0).astype(np.int8)   # Can be better
    df["VIP"]       = df["VIP"].apply(lambda x: 1 if "True" else 0)                         # Can be better

    # ----------------------------------------------------
    
    # TASK 05 - New Features
    
    # LuxaryCost
    df["LuxaryCost"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    
    
    # TASK 06 - Filling nan fields
    
    # It could be better
    simple_mean_colums = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for column in simple_mean_colums:
        df[column] = df[column].fillna(df[column].mean())
        
    df["LuxaryCost"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    
    df["Cabin_Num"] = df["Cabin_Num"].fillna(df["Cabin_Num"].mean())

    
    # HYPOTHESIS
    
    # H01
    if not test and hypothesis["H01_Age_Mean"]:
        tfalse_mean, ttrue_mean = df.groupby("Transported").mean()["Age"]
        tfalse_mean = round(tfalse_mean, 2)
        ttrue_mean = round(ttrue_mean, 2)
        df.loc[(df['Age'].isna()) & (df['Transported']), 'Age']  = ttrue_mean
        df.loc[(df['Age'].isna()) & (~df['Transported']), 'Age'] = tfalse_mean
    else:
        df["Age"].fillna(df['Age'].mean(), inplace=True)
        
    # H02
    if hypothesis["H02_Age_Groups"]:
        AgeGroup  = pd.DataFrame()
        AgeGroup["cut"]  = pd.cut(df['Age'], bins=8)   
        df["AgeGroups"] = AgeGroup["cut"]
        df = pd.get_dummies(df, prefix="AG", columns=["AgeGroups"])
        df.info()
        
    return df