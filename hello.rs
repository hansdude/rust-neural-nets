use std::f64;

struct Neuron {
    weights: Vec<f64>,
    value: f64
}

struct Entry {
    inputs: Vec<f64>,
    outputs: Vec<f64>
}

fn sigmoid(x: f64) -> f64 {
    let c = 1.0;
    1.0 / (1.0 + f64::exp(-c * x))
}
fn _sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
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
        vec![
            Neuron { weights: vec![0.0, 1.0, 1.0], value: 0.0 },
            Neuron { weights: vec![0.0, 1.0, 1.0], value: 0.0 },
        ],
        vec![
            Neuron { weights: vec![0.0, 1.0, 1.0], value: 0.0 },
        ],
    ];

    let input = vec![0.25, 0.25];
    forward(&input, &mut layers);

    println!("output");
    //for neuron in layers[1] {
        //println!("{}", neuron.value);
    //}
}

// dE/dw_l = sum(delta_i * ) *


// dE/dw_j = delta_i * h_j
// delta_i = (o_i - t) * o_i(1 - o_i)

// dE/dw_j = (o_i - t) * o_i(1 - o_i) * h_j
// dE/dw_j = dE/do_i + do_i/dn_i + dn_i/dw_j

// dn_i/dw_j = h_j
// n_i = W dot H
// do_i/dn_i = o_i(1 - o_i)
// o_i(n) = 1/(1+e^(-n_i))
// dE/do_i = o_i - t
// E = (1/2)sum((t - o_i)^2)

fn forward(input: &Vec<f64>, layers: &mut Vec<Vec<Neuron>>) {
    //layers.iter().fold(input, forward_layer)
    let mut current = input;
    for layer in layers {
        forward_layer(current, layer);
        current = &Vec::new();
        for neuron in layer {
            current.push(neuron.value)
        }
    }
}

fn forward_layer(input: &Vec<f64>, layer: &mut Vec<Neuron>) {
    for neuron in layer {
        neuron.value = sigmoid(
            neuron.weights.iter()
                .skip(1)
                .zip(input)
                .fold(
                    neuron.weights[0],
                    |acc, (weight, i)| weight.mul_add(*i, acc)));
    }
}

