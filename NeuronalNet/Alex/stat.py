# include data pipe
from dataloader import plot

for i in range(10):
	plot.plot_csv_multipath("models/model_0"+str(i)+"/plot.csv", figure="Model_"+str(i)).savefig("model"+str(i)+".png")


#from modelpipe import pipe
#from keras.utils import plot_model

#model = pipe.Pipe().get_model(model_name="models/model_01")
#plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)
#model = pipe.Pipe().get_model(model_name="models/model_02")
#plot_model(model, to_file='models/model_02/model.png', show_layer_names=True, show_shapes=True)
#model = pipe.Pipe().get_model(model_name="models/model_03")
#plot_model(model, to_file='models/model_03/model.png', show_layer_names=True, show_shapes=True)
#model = pipe.Pipe().get_model(model_name="models/model_04")
#plot_model(model, to_file='models/model_04/model.png', show_layer_names=True, show_shapes=True)
#model = pipe.Pipe().get_model(model_name="models/model_05")
#plot_model(model, to_file='models/model_05/model.png', show_layer_names=True, show_shapes=True)
#model = pipe.Pipe().get_model(model_name="models/model_06")
#plot_model(model, to_file='models/model_06/model.png', show_layer_names=True, show_shapes=True)