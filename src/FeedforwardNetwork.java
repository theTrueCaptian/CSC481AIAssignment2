
import java.util.ArrayList;


/**
 *
 * @author user
 */
public class FeedforwardNetwork {
    private ArrayList<Layer> networkLayers;

    //used to gather information from mainApp about network info
    private ArrayList<Integer> neuronCountLayer;
    
    public FeedforwardNetwork(){
        networkLayers = new ArrayList<Layer>();
        neuronCountLayer = new ArrayList<Integer>();
    }

    //simply gathers information on network structure
    public void addLayer(int neurons){
        neuronCountLayer.add(neurons);
    }

    //must be called after adding the layers in order for the network to be created
    public void create(){
        int i;
        for(i=0; i<neuronCountLayer.size()-1; i++){
            networkLayers.add(new Layer(neuronCountLayer.get(i), this, i, false));
        }
        networkLayers.add(new Layer(neuronCountLayer.get(i), this, i, true));
    }

    //used by the Layer object for getting the next layer
    public Layer getNextLayer(int currentLayer){
        if(currentLayer>=networkLayers.size()-1){
            return null;
        }
        return networkLayers.get(currentLayer+1);
    }

    //must be called after the network is created. this randomizes all the
    //weights by calling each layer's random function,
    public void randomize(){
        //dont randomize for the last layer
        for(int i=0; i<networkLayers.size()-1; i++){
            networkLayers.get(i).initWeights();
        }
    }

    //this is finally called to determine the output of the network given a set of inputs
    public double[] input(double inputSet[]){
        for(int i=0; i<networkLayers.size()-1; i++){ //go thru each network layer except for the last one
            if (networkLayers.get(i).isInputLayer()) {
                networkLayers.get(i).computeOutputs(inputSet);
            } else if (networkLayers.get(i).isHiddenLayer()) {

                networkLayers.get(i).computeOutputs(null);
            }
        }
        return networkLayers.get(networkLayers.size()-1).getOutput();
    }

    public void calcError(int layerIndex, double[] idealOutput){
        //clear all previous error data
        for(int i=0; i<networkLayers.size(); i++){
            networkLayers.get(i).clearError();
        }

        for(int i=networkLayers.size()-1; i>=0; i--){
            if(networkLayers.get(i).isOutputLayer()){
                //System.out.println("Calculating error for output layer:"+i);
                networkLayers.get(i).calcError(idealOutput);
            }else{
                //System.out.println("Calculating error for other layers:"+i);
                networkLayers.get(i).calcError();
            }
        }
    }

    public void learn(){
        //learn except for the last layer
        for(int i=0; i<networkLayers.size()-1; i++){
            //learning rate and momentum
            networkLayers.get(i).learn(.05, 0);
        }
        
    }
}
