import Layer from "./Layer";
import Neuron from "./Neuron";

class NeuralNetwork {
    layers: Layer[];
    trainingResult: number[][];
    constructor(layers: Layer[]) {
        this.layers = layers;
        this.trainingResult = [];
    }
    // next return layer if layer is the last of the neural network
    // and if its not the last that return the next layer
    next(layer: Layer) {
        return this.layers.indexOf(layer) === this.layers.length - 1 ?
            layer : this.layers[this.layers.indexOf(layer) + 1];
    }

    // this function return how much the neural network make error
    // its mean for MSE if we use it and the neural network return 4
    // but we expected 5 we will do 4-5 = -1 
    // -1^^2 == 1 => (1/n)*Loss == (1/1)*1 = 1
    // so the quantification of the error is 1
    // if we train the neural network on one value on the value we've seen
    errorValue(lossFunction: CallableFunction) {
        const results: Array<[number, number]> = [];
        //first element is the tab of entrys and the second is the expected result
        return (entrys: Array<[number[][], number]>) => {
            for (const [training, expect] of entrys) {
                const callBackEntrys: CallableFunction = this.layers[0].neuronsActivation();
                if(typeof callBackEntrys === 'function'){
                    results.push([callBackEntrys(training)(this), expect]);
                }else{
                    results.push([callBackEntrys, expect]);
                }
                this.empty();
            }
            this.trainingResult = results;
            return lossFunction(results);
        }
    }

    meanSquaredError(training: Array<[number, number]>) {
        let loss: number = 0;
        // this will do (predicted number - expected number) ^^ 2 to have the loss rate
        training.forEach(([predict, expect]) => {
            loss += Math.pow((predict - expect), 2);
        });
        // this return the average loss rate
        return (1 / training.length) * loss;
    }

    backTracking() {
        this.layers.forEach((layer, layerIndex) => {
            layer.neurons[0].forEach((neuron) => {
                if (neuron instanceof Neuron) {
                    if (layer === this.layers[this.layers.length - 1]) {
                        const lastNeuron = layer.neurons[0];
                        if (lastNeuron instanceof Neuron) {
                            lastNeuron.error = this.trainingResult[0][0] - this.trainingResult[0][0]
                        }
                    } else {
                        let errorSum = 0;
                        const nextLayer = this.layers[layerIndex + 1];

                        nextLayer.neurons.forEach((nextNeuron) => {
                            if (nextNeuron instanceof Neuron) {
                                const weight = nextNeuron.weight[layer.neurons.findIndex(([n]) => n === neuron)];
                                errorSum += nextNeuron.error * weight;
                            }
                        });

                        neuron.error = errorSum * neuron.sigmoidDerivative();
                    }
                    neuron.entrys.forEach((entry, index) => {
                        const weigthGradient = neuron.error * entry;
                        neuron.weight[index] -= neuron.learningRate * weigthGradient;
                    });
                    //for clarity
                    const biasGradient = neuron.error;
                    neuron.bias -= neuron.learningRate * biasGradient;

                }

            })
        })

    }

    empty(){
        for(const layer of this.layers){
            layer.empty();
        }
    }

}
export default NeuralNetwork; 