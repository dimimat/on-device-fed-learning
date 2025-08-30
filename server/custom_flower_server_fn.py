"""Flower server."""
import csv
import json
import sys
import os
import random
import struct # convert bytes to float32
import numpy as np
import pandas as pd
import subprocess
import concurrent.futures
import timeit
import time
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (Code,DisconnectRes,EvaluateIns,EvaluateRes,FitIns,FitRes,Parameters,ReconnectIns,Scalar,)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from history import History #from flwr.server.history import History -> there was some error with built in History class
from flwr.server.strategy import FedAvg, Strategy
from datetime import datetime

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

path_to_config = os.path.join(os.path.dirname(os.getcwd()),"config.txt")
#counter = 1

class CustomServer:
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        client_shift: int,
        strategy: Optional[Strategy] = None,
        model_config: list,
        filename: str,
        path_to_exp_directory: str,
        csv_filename: str,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.client_shift = client_shift
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.model_config = model_config
        self.filename = filename
        self.path_to_exp_directory = path_to_exp_directory
        self.csv_filename = csv_filename
        self.max_workers: Optional[int] = None
        self.num_rounds = None

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        # wait for N clients to simulate client dropout
        #self._client_manager.wait_for(5)
        # --------------------- main routine -------------------

        """Run federated averaging for a number of rounds."""
        history = History()
        if (self.client_shift == 0):
            #compile_model(model_config=self.model_config, fresh_copy=False)
            compile_model(model_config=self.model_config, fresh_copy=True) # make a clean copy of model
        else:
            #compile_model(model_config=self.model_config, fresh_copy=True) # make a clean copy of model
            print(f"client_shift: {self.client_shift}, YOU SHOULD COMPILE WITH UPDATED WEIGHTS")
            compile_model(model_config=self.model_config, fresh_copy=False)

        log(INFO, "FL starting")

        # ------------- Send data before training -----------
        self.send_data(filename=self.filename, timeout=timeout)
        #start_time = timeit.default_timer()

        # ----------------- Fed training routine -------------
        # Create the directory if it doesn't exist
        os.makedirs(self.path_to_exp_directory, exist_ok=True)
        delete_csv_file(self.path_to_exp_directory, self.csv_filename) # make sure there is not csv file with current experiments name
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            start_train = time.time()
            #print("ENTERED FIT ROUND")
            res_fit = self.fit_round(server_round=current_round,timeout=timeout)
            #print("EXIT FIT ROUND")
            end_train = time.time()
            if res_fit is not None:
                # findme
                # this is for the default aggregation
                parameters_prime, fit_metrics, results, failures  = res_fit
                #if parameters_prime:
                #    self.parameters = parameters_prime
                #    # update weights: save aggregated weights to file. then recompile model
                #    tensors = self.parameters.tensors
                #    save_weights(tensors)
                #    if num_rounds != 1:
                #        compile_model(model_config=self.model_config, fresh_copy=False)
                # this is for the custom aggregation

                tensors = parameters_prime
                if parameters_prime:
                   # update weights: save aggregated weights to file. then recompile model
                   save_weights(tensors)
                   #print("COMPILING MODEL WITH UPDATED WEIGHTS")
                   start_compile = time.time()
                   compile_model(model_config=self.model_config, fresh_copy=False)
                   end_compile = time.time()

                   # Store times
                   training_without_compile_times = end_train - start_train
                   #training_times = end_compile - start_compile
                   training_times = end_compile - start_train
                   compilation_time = end_compile - start_compile
                   print(f"Clean training time: {training_without_compile_times}  - Training with compilation time: {training_times}  - Compilation time: {compilation_time}")

                #tensors = parameters_prime
                #print(f"aggregated tensor: {tensors[-1]}")
           
            # Evaluate model (the global way)
            #print("ENTERED EVALUATE ROUND")
            evaluation_results = self.evaluate_round(server_round=current_round, timeout=timeout)
            #print("EXITING EVALUATION ROUND")
            if evaluation_results is not None:
                # note: in this round we should give the global trained model to clients for test phase
                # when we have common test we need only one client to take the test metrics
                evaluation_results_client_0 = evaluation_results[0]
                y, n_samples, accuracy, energy_before, energy_after, device_name, avg_rssi, latency, dl_speed, ul_speed, rx_data, tx_data = evaluation_results_client_0
                #y_true, y_pred = y[0], y[1]
                #print(f"type of y: {len(y)}")
                #print(f"n_samples: {n_samples}, accuracy: {accuracy}")

                # this is needed when there are multiple test sets (one for every client) 
                devices_names = []
                accuracy_bag = []
                energy_before_list = []
                energy_after_list = []
                y_true_list = []
                y_pred_list = []
                avg_rssi_list = []
                latency_list = []
                download_speed_list = []
                upload_speed_list = []
                rx_data_list = []
                tx_data_list = []
                for params, num_examples, acc, energy_before, energy_after, device_name, avg_rssi, latency, dl_speed, ul_speed, rx_data, tx_data in evaluation_results:
                    devices_names.append(device_name)
                    accuracy_bag.append(float(acc)) 
                    energy_before_list.append(energy_before)
                    energy_after_list.append(energy_after)
                    y_true, y_pred = params[0], params[1]
                    y_true_list.append(y_true)
                    y_pred_list.append(y_pred)

                    avg_rssi_list.append(avg_rssi)
                    latency_list.append(latency)
                    download_speed_list.append(dl_speed)
                    upload_speed_list.append(ul_speed)
                    rx_data_list.append(rx_data)
                    tx_data_list.append(tx_data)
                mean_accuracy = np.mean(accuracy_bag)

                log(INFO, "Global model accuracy on test set: %s", mean_accuracy)
                #print("global model accuracy on test set: %s", mean_accuracy)
                metrics = {
                    "n_samples": n_samples,
                    "y_true": y_true_list,
                    "y_pred": y_pred_list,
                    "accuracy": accuracy_bag,
                    "energy_before": energy_before_list,
                    "energy_after": energy_after_list,
                    "training_times": training_times,
                    "training_without_compile_times": training_without_compile_times,
                    "devices_names": devices_names,
                    "avg_rssi": avg_rssi_list,
                    "latency": latency_list,
                    "download_speed": download_speed_list,
                    "upload_speed": upload_speed_list,
                    "rx_data": rx_data_list,
                    "tx_data": tx_data_list,
                }

                add_evaluation_metrics_to_csv(current_round, self.path_to_exp_directory, self.csv_filename, **metrics)
                ## save attribute
                #file_path = "/Users/admin/Desktop/" + "experimentB13_2.txt"
                #with open(file_path, 'a') as file:  # 'a' mode appends to the file
                #    file.write(str(acc))
                #    file.write("\n")
            
        # get data 
        #end_time = timeit.default_timer()
        #elapsed = end_time - start_time
        #log(INFO, "FL training finished in %s", elapsed)

        return history

    def send_data(self,filename,timeout):
        # bypass evaluate path to send data to each client based on its id
        # [ (client,data_ins) , () , ... ()] . data_ins -> List[bytes(X_train) , ....]
        # maybe add send model functionalities. Send random data and random model for quick experiments
        client_instructions = configure_data_send(filename=filename,strategy=self.strategy,client_manager=self._client_manager,client_shift=self.client_shift,model_config=self.model_config)
        results, failures = evaluate_clients(client_instructions,max_workers=self.max_workers,timeout=timeout)

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        client_instructions = configure_evaluate(server_round=server_round,strategy=self.strategy,client_manager=self._client_manager)

        if not client_instructions:
            log(INFO, "configure_evaluate: no clients selected, skipping evaluation")
            return None
        log(
            INFO,
            "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        #print(f"results: {results[0]}, type of results: {type(results[0])}")
        #accuracy_aggreaged  = aggregate_evaluate(results)
        #return accuracy_aggreaged
        evaluation_results  = aggregate_evaluate(results)
        return evaluation_results


    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        client_instructions = configure_fit(server_round=server_round,strategy=self.strategy,client_manager=self._client_manager)

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round 
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
    
        # findme
        # client correctly returns the weights
        # but aggregation doubles them
        # check why
        decode_results(results)

        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        #aggregated_result: Tuple[
        #    Optional[Parameters],
        #    Dict[str, Scalar],
        #] = self.strategy.aggregate_fit(server_round, results, failures)  # <- aggregated weights

        #parameters_aggregated, metrics_aggregated = aggregated_result
        #return parameters_aggregated, metrics_aggregated, results, failures

        # custom aggregate fit
        parameters_aggregated = custom_aggregate_fit(results)
        return parameters_aggregated, "", "", "" 

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
     
    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }

        #print(f"submitted_fs : {submitted_fs}")
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(future=future, results=results, failures=failures)
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout) 
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result 

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)

def configure_fit(server_round: int, strategy: Strategy, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
    # change defaults configure fit functionality. a.k.a send model
    """Configure the next round of training."""
    config = {}
    config['operation_mode'] = 'train' # add a training tag
    if strategy.on_fit_config_fn is not None:
        # Custom fit config function provided
        config = strategy.on_fit_config_fn(server_round)

    model_name = "dnn.tflite"
    model = load_model(model_name)
    model2params = Parameters(tensors=[model], tensor_type="bytes/model") # convert model 2 parameters
    fit_ins = FitIns(model2params,config)

    # Sample clients
    sample_size, min_num_clients = strategy.num_fit_clients(client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

    # Return client/config pairs
    return [(client, fit_ins) for client in clients]


def configure_evaluate(server_round: int, strategy: Strategy, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
    """Configure the next round of inference."""
    config = {}
    config['operation_mode'] = 'inference' # add an inference tag
    
    model_name = "dnn.tflite"
    model = load_model(model_name)
    model2params = Parameters(tensors=[model], tensor_type="bytes/model") # convert model 2 parameters
    fit_ins = FitIns(model2params,config)

    # Sample clients
    sample_size, min_num_clients = strategy.num_evaluation_clients(client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

    # Return client/config pairs
    return [(client, fit_ins) for client in clients]


def load_model(model_name):
    path_to_model = os.path.join(os.path.dirname(os.getcwd()), "model_factory", "models", model_name)
    with open(path_to_model, "rb") as file:
        model = file.read()  # Read model as a whole file
        if not model:
            log(WARN,"model not loaded properly")
    #log(INFO,f"model type: {type(model)} size: {len(model)}") # same as self.parameters
    return model


def tofloat(tensor):
    format_string = '<f' # for little-endian
    weights = []
    #print("----------------------")
    for b1,b2,b3,b4 in zip(*[iter(tensor)]*4):
        # convert int to bytes data structure
        byte1 = bytes([b1])
        byte2 = bytes([b2])
        byte3 = bytes([b3])
        byte4 = bytes([b4])

        byte_sequence = byte1 + byte2 + byte3 + byte4 # we use float32 -> 4 bytes
        value = struct.unpack(format_string, byte_sequence)[0]
        #print(value)
        weights.append(value)
    return weights

def save_weights(tensors):
    # decompile tensors to (kernel,bias) format
    weight_list = []
    for kernel,bias in zip(*[iter(tensors)]*2): # expect { (kernel,bias) , .... , (kernel,bias) }
        # findme
        # our custom aggregation returns the values as float64
        # we do not need the byte2float conversion
        #kernel = tofloat(kernel)
        #bias = tofloat(bias)
        kernel = kernel
        bias = bias
        weight_list.append(kernel)
        weight_list.append(bias)
        #print(f"####### kernel length: {len(kernel)}")
        #print(f"####### bias length: {len(bias)}")

    # save weights on txt for model to be recompiled
    weights_name = "dnn.txt"
    #global counter
    #weights_name = "dnn_" + str(counter) + ".txt"
    #counter += 1
    path_to_weights = os.path.join(os.path.dirname(os.getcwd()), "model_factory", "weights", weights_name)
    # open file in write mode
    with open(path_to_weights, 'w') as f:
        for weights in weight_list:
            for value in weights:
                # temp to check validity of set weights
                #value = 0.101
                f.write(str(value))
                f.write("\t")
    
            f.write("\n")
    log(INFO,f"Weights saved, path to weights: {path_to_weights}")

def decode_results(results):
    #tensor = results[0][1] # FitRes
    weights = [
        (fit_res.parameters, fit_res.num_examples)
        for _, fit_res in results
    ]

    for w in weights:
        #print(f"n_examples: {w[1]}")
        tensors = w[0].tensors
        tensor_type = w[0].tensor_type
        #print(f"tensor {type(tensors)}")
        #print(f"tensor type {tensor_type}")
        #data = np.frombuffer(tensors, dtype=np.float32)
        for tensor in tensors:
            data = np.frombuffer(tensor, dtype=np.float32)
            #print(f"tensor shape: {data.shape}")

        #print("tensor")
        #print(np.frombuffer(tensors[5], dtype=np.float32))

#def custom_aggregate_fit(results):
#        if not results:
#            return None, {}
#
#        # Convert results
#        weights_results = [
#            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#            for client, fit_res in results
#        ]
#
#        # Extract weights and corresponding number of examples as separate arrays
#        weights, num_examples = zip(*weights_results)
#        num_examples = np.array(num_examples)  # Convert to NumPy array for broadcasting
#
#        # Calculate the total number of examples used during training
#        num_examples_total = np.sum(num_examples)
#
#        # Stack weights into a single NumPy array for vectorized operations
#        stacked_weights = [np.stack(client_weights) for client_weights in zip(*weights)]
#
#        # Compute the weighted sum for each layer
#        weights_prime = [
#            np.sum(layer * num_examples[:, np.newaxis], axis=0) / num_examples_total
#            for layer in stacked_weights
#        ]
#
#        return weights_prime
#        #return self.ndarrays_to_parameters(aggregate(weights_results)), {}

def custom_aggregate_fit(results):
    prob = 1.0
    if not results:
        return None, {}

    # 1 ─ Bernoulli sampling --------------------------------------------------
    selected = [
        (client, fit_res)
        for client, fit_res in results
        if random.random() < prob
    ]
    
    # Guarantee that at least ONE client is included
    if not selected:
        print(f"No client was selected. We want to guarantee that there is at least one")
        print(f"NUMBER OF CLIENTS: 0")
        return None
        #selected.append(random.choice(results))
    
    print(f"NUMBER OF CLIENTS: {len(selected)}")

    # Convert to (weights, num_examples) pairs
    weights_results = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in selected
    ]

    weights, num_examples = zip(*weights_results)
    num_examples = np.asarray(num_examples)

    # 3 ─ Weighted average ----------------------------------------------------
    num_examples_total = num_examples.sum()
    stacked_weights = [np.stack(layer) for layer in zip(*weights)]

    weights_prime = [
        (layer * num_examples[:, None]).sum(axis=0) / num_examples_total
        for layer in stacked_weights
    ]

    return weights_prime

def parameters_to_ndarrays(params):
    tensors = params.tensors
    return [np.frombuffer(tensor, dtype=np.float32) for tensor in tensors]

def compile_model(model_config,fresh_copy=False): # load model with init models or not
    script_name = "generate_dnn_model.py"
    path_to_script = os.path.join(os.path.dirname(os.getcwd()), "model_factory")

    #Helper: --learning_rate: float (default=0.01)  --n_inputs:int  --n_deep_layer: int  --n_outputs: int  init: bool (compile with init weights)
    #command = ["python", path_to_script, str(16), str(10), str(10), str(fresh_copy)]
    w1,w2,w3 = model_config[0], model_config[1], model_config[2] # get model weights
    init = 0
    if fresh_copy:
        init = 1
    command = ["python", script_name, str(w1), str(w2), str(w3), str(init)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=path_to_script) # run from scripts directory (because it uses relative paths for data saving)

    if fresh_copy == True:
        print("COMPILING MODEL WITH FRESH WEIGHTS")
        #log(INFO, "Compiling model with fresh weights, return code %s", str(result.returncode))
    else:
        print("COMPILING MODEL WITH AGGREGATED WEIGHTS")
        #log(INFO, "Compiling model with aggregated weights, return code %s", str(result.returncode))

    #print("########### Standard Output: #################" + str(result.stdout))
    #print("Standard Error:" + str(result.stderr))
    #print("########### Return Code: ################" + str(result.returncode))

def configure_data_send(filename,strategy,client_manager,client_shift,model_config): # [data(idx), data(idx), (.),......., (.)]
    # wait for clients
    sample_size, min_num_clients = strategy.num_fit_clients(client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
    config = {"n_inputs":model_config[0], "n_hidden":model_config[1] , "n_outputs":model_config[2]} # create model config
    #print(f"config: {config}") 

    client_instructions = []
    print(f"n_clients: {len(clients)}")
    for idx in range(len(clients)):
        client_id = idx + client_shift 
        data = load_data(client_id=client_id,filename=filename) # load data for client idx
        data2params = Parameters(tensors=data, tensor_type="bytes") # convert data to parameters
        data_ins = EvaluateIns(parameters=data2params,config=config)
        client_instructions.append((clients[idx],data_ins))
    
    return client_instructions

def load_data(client_id,filename):
    print(f"IDX: {client_id}")
    # get path from working data from config file
    with open(path_to_config, 'r') as file:
        path_to_working_data = file.readline().split(" ")[0]

    # load data and convert it to bytes
    X_train = pd.read_csv(path_to_working_data + "/" + filename + '/X_train/' + 'partition_' + str(client_id) + '.txt' , header=None ,delimiter=' ').values.astype(np.float32)
    y_train = pd.read_csv(path_to_working_data + "/" + filename + '/y_train/' + 'partition_' + str(client_id) + '.txt' ,header=None).to_numpy().flatten().astype('int32')
    X_test = pd.read_csv(path_to_working_data + "/" + filename + '/X_test/' + 'partition_' + str(client_id) + '.txt' , header=None , delimiter=' ').values.astype(np.float32)
    y_test = pd.read_csv(path_to_working_data + "/" + filename + '/y_test/' + 'partition_' + str(client_id) + '.txt' ,header=None).to_numpy().flatten().astype('int32')

    return [X_train.tobytes(),y_train.tobytes(),X_test.tobytes(),y_test.tobytes()]

def aggregate_evaluate(results: List[Tuple[ClientProxy, FitRes]]):
    evaluation_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, float(fit_res.metrics["accuracy"]), \
                    fit_res.metrics["energy_before"], fit_res.metrics["energy_after"], fit_res.metrics["device_name"], \
                    fit_res.metrics["avg_rssi"], fit_res.metrics["latency"], fit_res.metrics["download_speed"], \
                    fit_res.metrics["upload_speed"], fit_res.metrics["rx_data"], fit_res.metrics["tx_data"])
            for client, fit_res in results
    ]
    #accuracy_bag = [float(fit_res.metrics["accuracy"]) for client, fit_res in results]
    #return evaluation_results, np.mean(accuracy_bag)
    return evaluation_results

def delete_csv_file(path_to_csv_files, csv_filename):
    full_path = os.path.join(path_to_csv_files, csv_filename)
    if os.path.exists(full_path):
        os.remove(full_path)

def add_evaluation_metrics_to_csv(n_round, path_to_csv_files, csv_filename, **metrics):
    devices_names = metrics["devices_names"]
    n_samples = metrics["n_samples"]
    y_true_list = metrics["y_true"]
    y_pred_list = metrics["y_pred"]
    accuracy_list = metrics["accuracy"]
    energy_before_list = metrics["energy_before"]
    energy_after_list = metrics["energy_after"]
    training_times = metrics["training_times"]
    training_without_compile_times = metrics["training_without_compile_times"]
    avg_rssi_list = metrics["avg_rssi"]
    latency_list = metrics["latency"]
    download_speed_list = metrics["download_speed"]
    upload_speed_list = metrics["upload_speed"]
    tx_data_list = metrics["rx_data"]
    rx_data_list = metrics["tx_data"]

    energy_before_formatted = {
        device_name: energy_data
        for device_name, energy_data in zip(devices_names, energy_before_list)
    }

    energy_after_formatted = {
        device_name: energy_data
        for device_name, energy_data in zip(devices_names, energy_after_list)
    }

    accuracy_formatted = {
        device_name: accuracy 
        for device_name, accuracy in zip(devices_names, accuracy_list)
    }

    # Convert y_true and y_pred to lists (for JSON serialization)
    y_true_formatted = {
        device_name: y_true.tolist() if isinstance(y_true, np.ndarray) else y_true
        for device_name, y_true in zip(devices_names, y_true_list)
    }

    y_pred_formatted = {
        device_name: y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
        for device_name, y_pred in zip(devices_names, y_pred_list)
    }

    avg_rssi_formatted = {
        device_name: avg_rssi 
        for device_name, avg_rssi in zip(devices_names, avg_rssi_list)
    }

    latency_formatted = {
        device_name: latency
        for device_name, latency in zip(devices_names, latency_list)
    }

    download_speed_formatted = {
        device_name: download_speed
        for device_name, download_speed in zip(devices_names, download_speed_list)
    }

    upload_speed_formatted = {
        device_name: upload_speed
        for device_name, upload_speed in zip(devices_names, upload_speed_list)
    }

    tx_data_formatted = {
        device_name: tx_data
        for device_name, tx_data in zip(devices_names, tx_data_list)
    }

    rx_data_formatted = {
        device_name: rx_data
        for device_name, rx_data in zip(devices_names, rx_data_list)
    }

    # Writing to CSV
    full_path = os.path.join(path_to_csv_files, csv_filename)
    # Check if the file already exists and is not empty
    file_exists = os.path.isfile(full_path)
    is_empty = os.path.getsize(full_path) == 0 if file_exists else True
    with open(full_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header
        # mean accuracy is the mean of accuracy values from all the clients
        # whereas accuracy is from 1st client
        # these values will differ if we have mulitple test sets, one for every client
        # there is no need of storing y_true, y_pred for every client (we will only store them
        # for the 1st client
        if is_empty:
            writer.writerow(["n_round", "timestamp", "devices_names", "n_samples", "y_true", "y_pred", "accuracy", \
                "energy_before", "energy_after", "training_time", "training_without_compile_times", "avg_rssi", \
                "latency", "download_speed", "upload_speed", "rx_data", "tx_data"])
        
        # Write data (convert lists to JSON strings)
        timestamp = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
        #writer.writerow([n_round, timestamp, devices_names, n_samples, json.dumps(y_true.tolist()), json.dumps(y_pred.tolist()), \
        #        accuracy_formatted, energy_before_formatted, energy_after_formatted, training_times, training_without_compile_times])
        writer.writerow([n_round, timestamp, devices_names, n_samples, json.dumps(y_true_formatted), json.dumps(y_pred_formatted), \
                accuracy_formatted, json.dumps(energy_before_formatted), json.dumps(energy_after_formatted), training_times, \
                training_without_compile_times, json.dumps(avg_rssi_formatted), json.dumps(latency_formatted), json.dumps(download_speed_formatted), \
                json.dumps(upload_speed_formatted), json.dumps(tx_data_formatted), json.dumps(rx_data_formatted)])

def parse_energy_data(energy_data):
    energy_data = energy_data.strip("{} ")
    energy_dict = {}
    for item in energy_data.split(", "):
        key, value = item.split(" = ")
        energy_dict[key.strip()] = value.strip()
    return json.dumps(energy_dict)


