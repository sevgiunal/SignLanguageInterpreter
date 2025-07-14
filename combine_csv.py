import pandas as pd

# Load the first CSV file
df1 = pd.read_csv('./House-Sev-1711929550.csv')

# Load the second CSV file
df2 = pd.read_csv('./Hello-Sev-1711929771.csv')

df3 = pd.read_csv('./What-Sev-1711930322.csv')

df4 = pd.read_csv('./Day-Sev-1711931033.csv')

df5 = pd.read_csv('./Know-Sev-1711931493.csv')

df6 = pd.read_csv('./Hello-Melike-1711931931.csv')

df7 = pd.read_csv('./House-Melike-1711932169.csv')

df8 = pd.read_csv('./Know-Melike-1711932244.csv')

df9 = pd.read_csv('./Day-Melike-1711932467.csv')

df10 = pd.read_csv('./What-Melike-1711932038.csv')

df11 = pd.read_csv('./Hello-Ayse-1711933067.csv')

df12 = pd.read_csv('./House-Ayse-1711933653.csv')

df13 = pd.read_csv('./Know-Ayse-1711933555.csv')

df14 = pd.read_csv('./Day-Ayse-1711933774.csv')

df15 = pd.read_csv('./What-Ayse-1711933280.csv')

df16 = pd.read_csv('./Name-Sev-1711934394.csv')

df17 = pd.read_csv('./Name-Ayse-1711934622.csv')

df18 = pd.read_csv('./What-Sev-1711934826.csv')

df19 = pd.read_csv('./Name-Melike-1711970751.csv')

df20 = pd.read_csv('./Bus-Sev-1712149313.csv')

df21 = pd.read_csv('./Water-Sev-1712149719.csv')

df22 = pd.read_csv('./Please-Sev-1712150367.csv')

df23 = pd.read_csv('./Camera-Sev-1712151021.csv')

df24 = pd.read_csv('./Bus-Ilhami-1712151454.csv')

df25 = pd.read_csv('./Camera-Ilhami-1712151704.csv')

df26 = pd.read_csv('./Please-Ilhami-1712151845.csv')

df27 = pd.read_csv('./Water-Ilhami-1712151566.csv')

df28 = pd.read_csv('./Bus-Nezahat-1712152568.csv')

df29 = pd.read_csv('./Camera-Nezahat-1712152982.csv')

df30 = pd.read_csv('./Water-Nezahat-1712152777.csv')

df31 = pd.read_csv('./Please-Nezahat-1712153056.csv')

# Concatenate them row-wise
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28, df29, df30, df31], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('./combined_dataset.csv', index=False)
