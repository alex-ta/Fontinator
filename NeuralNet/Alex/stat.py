from dataloader import plot
# plot epoch csv as image
for i in range(10):
	plot.plot_csv_multipath("models/model_0"+str(i)+"/plot.csv", figure="Model_"+str(i)).savefig("model"+str(i)+".png")


	
# Plot Model as Image	
#from modelpipe import pipe
#from keras.utils import plot_model
#for i in range(10):
#	plot.plot_csv_multipath("models/model_0"+str(i)+"/plot.csv", figure="Model_"+str(i)).savefig("model"+str(i)+".png")
#	model = pipe.Pipe().get_model(model_name="models/model_0"+str(i))
#	plot_model(model, to_file="models/model_0"+str(i)+"model.png", show_layer_names=True, show_shapes=True)

