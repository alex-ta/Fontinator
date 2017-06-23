from modelpipe import pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_1", train_size=0.6)
datapipe.load_data(flatten=0, print_out=1)
for i in range(10):
	result = datapipe.eval(model_name = "models/model_0"+str(i), batch_size = 100)