import pandas as pd
import flwr as fl
import binary_model_training as bmt

vulnerability_df = pd.read_pickle("proccessed_dataset_for_analysis.pickle")

vulnerability_df = vulnerability_df[:int(len(vulnerability_df.index)*1/4)]

vulnerability_df.Vulnerability_status.value_counts()

model,early_stopping,X_train,y_train,X_test,y_test = bmt.train_model(vulnerability_df)


class FederatedBinaryClient(fl.client.NumPyClient):
    # def get_parameters(self):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        # model_history = model.fit(X_train, y_train, epochs=1000,callbacks=early_stopping, validation_data=(X_test,y_test))
        model_history = model.fit(X_train, y_train, epochs=3,callbacks=early_stopping, validation_data=(X_test,y_test))
        print(model.summary())

        print(model_history.history.keys())
        return model.get_weights(),len(X_train),{}

    def evaluate(self, parameters,config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test),{"accuracy":accuracy}


fl.client.start_numpy_client(server_address="192.168.1.4:8080", client=FederatedBinaryClient())