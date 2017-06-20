# include data pipe
from dataloader import plot

plot.plot_csv_multipath("models/model_01/plot.csv", figure="Model_1").savefig("model1.png")
plot.plot_csv_multipath("models/model_02/plot.csv", figure="Model_2").savefig("model2.png")
plot.plot_csv_multipath("models/model_03/plot.csv", figure="Model_3").savefig("model3.png")
plot.plot_csv_multipath("models/model_04/plot.csv", figure="Model_4").savefig("model4.png")
plot.plot_csv_multipath("models/model_05/plot.csv", figure="Model_5").savefig("model5.png")
plot.plot_csv_multipath("models/model_07/plot.csv", figure="Model_7").savefig("model7.png")
plot.plot_csv_multipath("models/model_08/plot.csv", figure="Model_8").savefig("model8.png")
plot.plot_csv_multipath("models/model_09/plot.csv", figure="Model_9").savefig("model9.png")
plot.plot_csv_multipath("models/model_10/plot.csv", figure="Model_10").savefig("model10.png")
plt = plot.plot_csv_multipath("models/model_06/plot.csv", figure="Model_6")
plt.savefig("model6.png")
plt.show();


from modelpipe import pipe
from keras.utils import plot_model

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