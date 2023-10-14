from circuit_manipulation import *
import numpy as np

def getCliffordLoss(train_x,train_y,inputs,encoder,classifier):
    """
    Obtain the (exact) Cliffordized loss for specified hypermapper inputs
    
    train_x (Iterable[Iterable[Float]]): The coordinates for the training data
    train_y (Iterable[Float]): The classifications
    inputs (Dict): Dictionary of inputs in the format that hypermapper uses 
    encoder (QuantumCircuit): The data encoding circuit (feature mapper)
    classifier (QuantumCircuit): The classifier portion of the circuit
    Returns:
    (Float): Cross entropy loss
    """
    softening = 1e-7 #create a softening parameter to avoid divergences
    parameters = np.array(list(inputs.values()))*np.pi/2
    qc_classifier = classifier.assign_parameters(parameters)
    loss=0
    for i,x in enumerate(train_x):
        temp_encoder = Cliffordize(encoder.assign_parameters(x))
        qc = temp_encoder.compose(qc_classifier)
        trans_qc = transform_to_allowed_gates(qc)
        stim_circuit = qiskit_to_stim(trans_qc)
        sim = stim.TableauSimulator()
        sim.do_circuit(stim_circuit)
        expect = sim.peek_observable_expectation(stim.PauliString('Z'*encoder.num_qubits))
        probabilities = {-1: (1-expect)/2,1: (1+expect)/2}
        y=train_y[i]
        p_correct = probabilities[y]
        p_incorrect = probabilities[-y]
        loss+= ((p_correct-1)**2+p_incorrect**2)
    return loss

import hypermapper
import json
import sys
from numbers import Number

def BayesianOptimizer(fun,num_params,iterations,save_dir,name):
    """
    General purpose hypermapper based optimizer
    (Function) fun: Function to optimize over
    (Int) num_params: The number of parameters
    (Int) iterations: The number of iterations
    (String) save_dir: text name of the save directory
    (String) name: name for all the logging
    """
    
    hypermapper_config_path = save_dir + "/"+name+"_hypermapper_config.json"
    config = {}
    config["application_name"] = "cafqa_optimization_"+name
    config["optimization_objectives"] = ["value"]
    config["design_of_experiment"] = {}
    config["design_of_experiment"]["number_of_samples"] = iterations
    config["optimization_iterations"] = iterations
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = True
    config["print_posterior_best"] = True
    for i in range(num_params):
        x = {}
        x["parameter_type"] = "ordinal"
        x["values"] = [0, 1, 2, 3]
        x["parameter_default"] = 0
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] = save_dir + '/'+name+'_hypermapper_log.log'
    config["output_data_file"] = save_dir + "/"+name+"_hypermapper_output.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    stdout=sys.stdout
    with open(save_dir+"/"+name+'_optimizer_log.txt', 'w') as sys.stdout:
        hypermapper.optimizer.optimize(hypermapper_config_path,fun)
    sys.stdout = stdout
    
    fun_ev = np.inf
    x = None
    with open(config["log_file"]) as f:
        lines = f.readlines()
        counter = 0
        for idx, line in enumerate(lines[::-1]):
            if line[:16] == "Best point found" or line[:29] == "Minimum of the posterior mean":
                counter += 1
                parts = lines[-1-idx+2].split(",")
                value = float(parts[-1])
                if value < fun_ev:
                    fun_ev = value
                    x = [int(y) for y in parts[:-1]]
            if counter == 2:
                break
    return fun_ev, x

def VQC_optimize(train_x,train_y,classifier,encoder,iterations,save_dir,name):
    """
    Run a hypermapper optimization for VQC parameters
    train_x (Iterable[Iterable[Float]]): The coordinates for the training data
    train_y (Iterable[Float]): The classifications
    encoder (QuantumCircuit): The data encoding circuit (feature mapper)
    classifier (QuantumCircuit): The classifier portion of the circuit
    (Int) iterations: The number of iterations
    (String) save_dir: text name of the save directory
    (String) name: name for all the logging
    """
    num_params = classifier.num_parameters
    result = BayesianOptimizer(fun = lambda inputs: getCliffordLoss(train_x=train_x,train_y=train_y,inputs=inputs,encoder=encoder,classifier=classifier) ,
                               num_params=num_params,
                               iterations=iterations,
                               save_dir=save_dir,
                               name=name)
    
    return result