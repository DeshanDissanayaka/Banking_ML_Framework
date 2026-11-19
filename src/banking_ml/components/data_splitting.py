from sklearn.model_selection import train_test_split

class DataSplitting:
    @staticmethod
    def split(X, y, test_size=0.2, validation_size=0.2, random_state=42):
        # Baseline only. For time-dependent banking data, implement a chronological
        # split; for repeated customers, consider group-aware splitting.
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        relative_val = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=relative_val,
            random_state=random_state, stratify=y_trainval
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
