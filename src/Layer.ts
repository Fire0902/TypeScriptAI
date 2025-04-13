import NeuralNetwork from "./NeuralNetwork";
import Neuron from "./Neuron";

class Layer {

    static neuralNetwork: NeuralNetwork;

    // neurons represent a tuple where the first element is the neuron, 
    // the second all the entry this neuron gonna receive
    // and the third is the mask to know if the element is usefull or not

    neurons: Array<[Neuron, number[], boolean[]]>;

    // activationResults represent the result of all the neurons activation function

    activationResults: number[];

    constructor(neurons: Neuron[]) {
        this.neurons = neurons.map(neuron => [neuron, [], []])
        this.activationResults = [];
    }

    static setNeuralNetwork(neuralNetwork: NeuralNetwork) {
        Layer.neuralNetwork = neuralNetwork;
    }

    setEntrys(entrys: Array<number[]>) {
        this.neurons.forEach((element, index) => {
            // if the length of the weigth array is bigger than the entrys at the index of the neuron
            // we fill with all the entry avaible and the emptys element we fill them with 0 and put a mask
            // to say to the neuron this entry is useless
            if (element[0].weight.length > entrys[index].length) {

                entrys[index].forEach(entry => {
                    element[1].push(entry);
                    element[2].push(true);
                });

                const restLength = element[0].weight.length - entrys[index].length;

                for (let i = 0; i < restLength; i++) {
                    element[1].push(0);
                    element[2].push(false);
                }

            }
            // else we just fill all the entry into the neurons element and if the entrys are bigger  
            // we ignore the overflow of entry
            else {
                element[1] = entrys[index]
                element[2] = entrys[index].map(() => true);
            }
        })
    }

    // this function will activate all the neurons and if setEntrys was not call before
    // it will return a callback that take in argument the entrys of the neuronActivation
    neuronsActivation() {
        for (const neuron of this.neurons) {
            const entrys = neuron[1];

            const actualNeuron = neuron[0];

            const masks = neuron[2]

            if (entrys.length !== 0) {
                this.activationResults.push(actualNeuron.weightedSum(entrys, masks));
            }
            else return (entrys: Array<number[]>) => {
                this.setEntrys(entrys);
                return this.neuronsActivation();
            };
        }
        // neuralNetwork.next return this if this is the last layer 
        if (Layer.neuralNetwork) {
            const nextLayer: Layer = Layer.neuralNetwork.next(this);
            // it will call the nextLayer until its the last one
            if (nextLayer === this) {
                // we return the first element because the last layer must have one neuron so one result
                return this.activationResults[0];
            } else {
                type NeuronCallback = (entrys: number[][]) => any;
                // we call the next layer neuronsActivation and we use the callback to pass the entrys
                const nextLayerResult : NeuronCallback = nextLayer.neuronsActivation();
                if (typeof nextLayerResult === 'function') {
                    return nextLayerResult(nextLayer.neurons.map(() => this.activationResults));
                } else {
                    return nextLayerResult;
                }
            }
        } else return (nn: NeuralNetwork) => {
            Layer.setNeuralNetwork(nn);
            this.neuronsActivation();
        }

    }

    empty() {
        for (const neuron of this.neurons) {
            neuron[0].empty();
        }
    }

}
export default Layer;