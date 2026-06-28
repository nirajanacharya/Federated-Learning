def basic_eda(df, logger):
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts().to_string()}")
    missing = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing}")
