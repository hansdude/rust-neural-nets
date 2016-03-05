use std::f64;
use std::iter;
use std::mem;

struct Neuron {
    weights: Vec<f64>,
    new_weights: Vec<f64>,
    output: f64,
    delta: f64,
}

impl Neuron {
    fn new(weights : Vec<f64>) -> Neuron {
        Neuron { weights: weights, new_weights: vec![], output: 0.0, delta: 0.0 }
    }

    fn compute_delta(&mut self, errorFromOutput: f64) {
        self.delta = errorFromOutput * self.output * (1.0 * self.output)
    }

    fn compute_new_weights(&mut self, learningRate : f64, inputs: Vec<f64>) {
        let delta = self.delta;
        self.new_weights = iter::once(1.0).chain(inputs)
            .zip(self.weights.to_owned())
            .map(|(weight, input)| weight + learningRate * delta * input)
            .collect();
    }

    fn use_new_weights(&mut self) {
        self.weights = mem::replace(&mut self.new_weights, vec![]);
    }

    fn error_from_inputs(&self) -> Vec<f64> {
        let delta = self.delta;
        self.weights.iter().skip(1).map(|weight| weight * delta).collect()
    }
}

struct Layer {
    inputs: Vec<f64>,
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(neurons: Vec<Neuron>) -> Layer {
        Layer { neurons: neurons, inputs: vec![] }
    }

    fn outputs(&self) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.output).collect()
    }

    fn error_from_inputs(&self) -> Vec<f64> {
        let errors : Vec<Vec<f64>> = self.neurons.iter().map(|neuron| neuron.error_from_inputs()).collect();
        (0..errors[0].len()).map(|i| errors.iter().fold(0.0, |acc, err| acc + err[i])).collect()
    }
}

struct Entry {
    inputs: Vec<f64>,
    outputs: Vec<f64>,
}

fn sigmoid(x: f64) -> f64 {
    let c = 1.0;
    1.0 / (1.0 + f64::exp(-c * x))
}

fn main() {
    // XOR function
    let training = vec![
        Entry {
            inputs: vec![0.0, 0.0],
            outputs: vec![0.0],
        },
        Entry {
            inputs: vec![0.0, 1.0],
            outputs: vec![1.0],
        },
        Entry {
            inputs: vec![1.0, 0.0],
            outputs: vec![1.0],
        },
        Entry {
            inputs: vec![1.0, 1.0],
            outputs: vec![0.0],
        },
    ];

    let mut layers = vec![
        Layer::new(vec![
            Neuron::new(vec![0.0, 1.0, 1.0]),
            Neuron::new(vec![0.0, 1.0, 1.0]),
        ]),
        Layer::new(vec![
            Neuron::new(vec![0.0, 1.0, 1.0]),
        ]),
    ];

    for entry in training {
        forward(&entry.inputs, &mut layers);
        backward(-0.001, &entry, &mut layers);
    }

    println!("output");
}

fn far(learningRate : f64, entry: &Entry, layers: &mut Vec<Layer>) {
    let mut errorFromOutput : Vec<f64> =
        (&layers[0]).outputs().iter().zip(&entry.outputs)
            .map(|(out, target)| out - target)
            .collect();

    //(&mut layers[0]).neurons[0].compute_delta(0.0);
    //for layer in layers.iter().rev() {
        //for (neuron, efo) in (&mut layer.neurons).iter().zip(&errorFromOutput) {
            //neuron.compute_delta(*efo);
        //}
    //}

    // compute_delta for each neuron in final layer
    // compute_new_weights for each neuron in final layer
    // calculate errorFromOutput for ea
}

fn backward(learningRate : f64, entry: &Entry, layers: &mut Vec<Layer>) {
    let layer = (&layers[0].neurons).to_owned();

    let outputs : Vec<f64> = layer.iter().map(|neuron| neuron.output).collect();
    let dError_dOut : Vec<f64> = entry.outputs.iter()
        .zip(&outputs)
        .map(|(expected, actual)| actual - expected)
        .collect();
    let dOut_dNet : Vec<f64> = outputs.iter().map(|out| out * (1.0 - out)).collect();
    let deltas : Vec<f64> = dOut_dNet.iter().zip(dError_dOut).map(|(dedo, dodn)| dedo * dodn).collect();
    /*let newWeights : Vec<Vec<f64>> =
        deltas.enumerate().map(|(i, delta)|
            layer.weights.iter().skip(1).map(|weight|

                ));
                */
}

fn foo(learningRate : f64, nodeInputs : Vec<f64>, weights: Vec<f64>, output : f64, errorFromOutput : f64) -> (f64, Vec<f64>) {
    let delta = errorFromOutput * output * (1.0 - output);
    (0.0, vec![])
}

fn compute_new_weights_for_node(delta : f64, learningRate : f64, nodeInputs : Vec<f64>, weights: Vec<f64>) -> Vec<f64> {
    iter::once(1.0).chain(weights)
        .zip(nodeInputs)
        .map(|(weight, nodeInput)| weight + learningRate * delta * nodeInput)
        .collect()
}

fn forward(input: &Vec<f64>, layers: &mut Vec<Layer>) -> Vec<f64> {
    let mut current = input.to_owned();
    for layer in layers {
        forward_layer(current, &mut layer.neurons);
        current = layer.outputs();
    }
    current
}

fn forward_layer(input: Vec<f64>, layer: &mut Vec<Neuron>) {
    for neuron in layer {
        neuron.output = sigmoid(
            neuron.weights.iter()
                .skip(1)
                .zip(&input)
                .fold(
                    neuron.weights[0],
                    |acc, (weight, i)| weight.mul_add(*i, acc)));
    }
}

