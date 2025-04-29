import numpy as np

def apply_scaling(df, scaling_params, trigono_columns):
    df_scaled = df.copy()
    for column in df.columns:
        if column in trigono_columns:
            df_scaled[column] = (df_scaled[column] + 1.0) / 2.0
        elif column == 'relativeStartTime':
            df_scaled[column] = df_scaled[column] / scaling_params[column]['max']
        elif column == 'stripeSize':
            # fallback scaling for stripeSize using train rank distribution
            df_scaled['stripeSizeRank'] = df_scaled['stripeSize'].rank(method='average')
            df_scaled['stripeSize'] = (df_scaled['stripeSizeRank'] - 1) / (scaling_params[column]['denom'])
            df_scaled.drop('stripeSizeRank', axis=1, inplace=True)
        else:
            df_scaled[column] = np.log(df_scaled[column] + 0.01)
            df_scaled[column] = (df_scaled[column] - scaling_params[column]['min']) / (scaling_params[column]['max'] - scaling_params[column]['min'])
    return df_scaled