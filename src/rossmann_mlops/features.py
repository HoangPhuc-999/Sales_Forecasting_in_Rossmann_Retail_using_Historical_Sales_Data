import pandas as pd
import numpy as np


# ===== ROW-LEVEL FEATURE =====
def extract_row_logic(df):
    df = df.copy()

    # ===== DATE FEATURES =====
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.weekday + 1

    # ===== HOLIDAY =====
    holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    df['StateHoliday'] = df['StateHoliday'].astype(str).map(holiday_map).fillna(0).astype(int)

    # ===== PROMO TIME =====
    sales_weeks = df['Year'] * 52 + df['WeekOfYear']
    promo_weeks = df['Promo2SinceYear'] * 52 + df['Promo2SinceWeek']
    df['Promo2Open_Month'] = (sales_weeks - promo_weeks) / 4.0

    # ===== COMPETITION TIME =====
    sales_months = df['Year'] * 12 + df['Month']
    comp_months = df['CompetitionOpenSinceYear'] * 12 + df['CompetitionOpenSinceMonth']
    df['CompetitionOpen_Month'] = sales_months - comp_months

    # ===== FIX LOGIC =====
    df.loc[df['Promo2'] == 0, 'Promo2Open_Month'] = 0
    df.loc[df['Promo2SinceYear'] == 0, 'Promo2Open_Month'] = 0
    df.loc[df['CompetitionOpenSinceYear'] == 0, 'CompetitionOpen_Month'] = 0

    # ===== CLIP =====
    df['Promo2Open_Month'] = df['Promo2Open_Month'].clip(0, 24)
    df['CompetitionOpen_Month'] = df['CompetitionOpen_Month'].clip(0, 24)

    # ===== PROMO INTERVAL =====
    month_map = {
        1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'
    }

    df['month_tmp'] = df['Month'].map(month_map)
    df['Is_Promo2_Month'] = 0

    mask = (df['Promo2'] == 1) & (df['PromoInterval'] != 'None') & (df['PromoInterval'] != '')

    df.loc[mask, 'Is_Promo2_Month'] = df.loc[mask].apply(
        lambda x: 1 if x['month_tmp'] in x['PromoInterval'] else 0, axis=1
    )

    df.drop(columns=['month_tmp'], inplace=True)

    # ===== STORE ENCODING =====
    store_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assort_map = {'a': 0, 'b': 1, 'c': 2}

    df['StoreType'] = df['StoreType'].map(store_map).astype(int)
    df['Assortment'] = df['Assortment'].map(assort_map).astype(int)

    return df


# ===== MAIN FEATURE PIPELINE =====
def run_feature_engineering(train_merged, test_merged):

    # ===== DROP EARLY (LEAKAGE + REDUNDANT) =====
    train_merged = train_merged.drop(columns=['Customers', 'Open'], errors='ignore')
    test_merged = test_merged.drop(columns=['Customers', 'Open'], errors='ignore')

    # ===== APPLY ROW LOGIC =====
    train_processed = extract_row_logic(train_merged)
    test_processed = extract_row_logic(test_merged)

    # ===== DROP USELESS =====
    columns_to_drop = [
        'Date',
        'CompetitionDistance',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2SinceWeek',
        'Promo2SinceYear',
        'PromoInterval'
    ]

    train_final = train_processed.drop(columns=columns_to_drop, errors='ignore')
    test_final = test_processed.drop(columns=columns_to_drop, errors='ignore')

    # ===== FINAL CLEAN  =====
    train_final.drop(columns=['Sales'], inplace=True, errors='ignore')
    test_final.drop(columns=['Open'], inplace=True, errors='ignore')

    return train_final, test_final
def main():

    # ===== LOAD DATA TỪ PROCESSING =====
    train = pd.read_csv("train_processed.csv")
    test = pd.read_csv("test_processed.csv")

    # ⚠️ convert lại datetime (bắt buộc)
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

    # ===== RUN FEATURE =====
    train_final, test_final = run_feature_engineering(train, test)

    print("Train final:", train_final.shape)
    print("Test final:", test_final.shape)

    # ===== SAVE FILE =====
    train_final.to_csv("train_final.csv", index=False)
    test_final.to_csv("test_final.csv", index=False)

    print("Saved feature data!")


if __name__ == "__main__":
    main()