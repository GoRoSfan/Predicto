import json

from pprint import pprint
import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV


def title_to_embedding(title, embedding='all-MiniLM-L6-v2'):
    embd_model = SentenceTransformer(embedding)
    return embd_model.encode(title)


class PredictoModel:
    def __init__(self, model_path: str = 'model', data_file=None):
        if data_file is not None:
            self.df = pd.read_json(data_file, encoding="utf-8")
            self.df = self.preprocess_data(self.df)
        else:
            self.df = None

        self.model_path = model_path
        try:
            self._model = keras.models.load_model(model_path)
        except OSError:
            self._model = None

    @property
    def model(self):
        if self._model is None:
            raise ValueError("No model available.")
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._model.save(self.model_path)

    @staticmethod
    def preprocess_data(df):
        mean_df = np.mean(df.claps)
        df = df[df.claps < mean_df * 40]

        model_params = {
            "min": int(np.min(df.claps)),
            "max": int(np.max(df.claps))
        }
        PredictoModel.save_model_params(model_params)
        df["claps"] = (df.claps - model_params["min"]) / (model_params["max"] - model_params["min"])
        return df

    @staticmethod
    def save_model_params(model_params):
        model_params = json.dumps(model_params)
        with open("model_params.json", "w") as f:
            f.write(model_params)

    @staticmethod
    def get_model_params():
        with open("model_params.json") as f:
            return json.load(f)

    def update_dataset_with_embeddings(self, file):
        file_name, file_type = file.split(".")
        if file_type == "csv":
            self.df = pd.read_csv(file, encoding="utf-8")
        elif file_type == "json":
            self.df = pd.read_json(file, encoding="utf-8")

        self.df["embeddings"] = self.df.title.apply(lambda x: title_to_embedding(x))
        self.df.to_json(f'updated_{file_name}.json')

    def new_model(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=(384,)))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer="adam",
                      loss="mse",
                      metrics=['mae', 'accuracy'])

        model.summary()
        self.model = model

    def update_model(self, file_name=None):
        if file_name is not None:
            self.update_dataset_with_embeddings(file_name)
            self.preprocess_data(self.df)

        X = np.array([np.asarray(embd).astype('float32') for embd in self.df.embeddings])
        y = np.asarray(self.df.claps).astype('float32')

        self.model.fit(X, y, epochs=20, batch_size=256, validation_split=0.2)
        self.model.save(self.model_path)

    def predict_claps(self, title: str):
        embd = np.array([title_to_embedding(title), title_to_embedding(title)])
        model_params = self.get_model_params()
        predicts = [p * (model_params["max"] - model_params["min"]) + model_params["min"] for p in
                    self.model.predict(embd)]
        return predicts


def evaluate(predictions, test_labels):
    errors = abs(predictions - test_labels)
    print('Model Performance')
    print('Average claps: {:0.1f}'.format(np.mean(predictions)))
    print('Max claps: {:0.1f}'.format(np.max(predictions)))
    print('Average Error: {:0.2f}'.format(np.mean(errors)))


class PredictoRFR:
    @staticmethod
    def get_best_params_by_random(X_train, y_train):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        pprint(random_grid)
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)
        rf_random.fit(X_train, y_train)
        pprint(rf_random.best_params_)

    @staticmethod
    def get_best_params_by_grid(df):
        X_train, X_test, y_train, y_test = train_test_split(
            list(df.embeddings.values),
            df.claps,
            test_size=0.1,
            random_state=42
        )
        param_grid = {
            'bootstrap': [True],
            'max_depth': [10, 20, 40],
            'max_features': [2, 3, 4],
            'min_samples_leaf': [4],
            'min_samples_split': [2, 4, 8],
            'n_estimators': [500, 800, 1000]
        }

        # Create a base model
        rf = RandomForestRegressor(random_state=42)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2, return_train_score=True)

        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        preds = best_grid.predict(X_test)
        evaluate(preds, y_test)

    @staticmethod
    def new_rfr(df):
        rfr = RandomForestRegressor(n_estimators=500, bootstrap=True, max_features=5, min_samples_leaf=4,
                                    min_samples_split=2, max_depth=20)
        rfr.fit(list(df.embeddings.values), df.claps)

    @staticmethod
    def predict_claps(titles):
        embds = title_to_embedding(titles, embedding='all-MiniLM-L6-v2')
        rfr = joblib.load("random_forest.joblib")
        return rfr.predict(embds)


if __name__ == '__main__':
    pass
    pm = PredictoModel(data_file="updated_train_set.json")
    # pm.new_model()
    # pm.update_model()
    print(pm.predict_claps("something for testing"))
    #
    # test_df = pd.read_json("updated_medium_data.json")
    # test_df = pm.preprocess_data(test_df)
    # print(np.mean(test_df.claps))
    # print(np.max(test_df.claps))
    # predictions = [l[0] for l in pm.predict_claps(list(test_df.title))]
    # print(predictions)
    # evaluate(np.array(predictions), test_df.claps)
