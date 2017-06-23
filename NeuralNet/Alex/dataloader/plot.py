import matplotlib.pyplot as plt
import csv

default_path = "result.csv"
default_color = ["red","blue","yellow","green","grey","blue"]

def write_csv(var, path = default_path):
    keys = list(var.keys())
    values = list(var.values())
    with open(path, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',quotechar=';', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(keys)
        for vi in range(len(values[0])):
            row = []
            for ki in range(len(keys)):
                row.append(values[ki][vi])
            writer.writerow(row)

def write_line_csv(row, path=default_path):
	with open(path, 'a', newline='\n') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar=';', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(row)

def read_csv(path = default_path):
    keys = []
    values = []
    with open(path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar=';')
        for row in reader:
            if not len(keys):
                keys = row
                for key in keys:
                    values.append([])
            else:
                for ki in range(len(keys)):
                    values[ki].append(row[ki])
    return keys, values

def plot_csv(path = default_path):
    keys, values = read_csv(path)
    for ki in range(len(keys)):
        plt.plot(values[ki], range(len(values[ki])), color=default_color[ki], linewidth=2.5, linestyle="-", label=keys[ki])
    plt.legend(loc='upper left')
    plt.show()
	
def plot_csv_multipath(path = default_path, skeys=["acc","val_acc","loss","val_loss","time"], figure="Figure_Name", group=2):
    plt.figure(figure)
    keys, values = read_csv(path)
    i = 0
    plt.subplot(221)
    for ki in range(len(skeys)):
        i = i + 1
        vi = keys.index(skeys[ki])
        plt.plot(range(len(values[vi])), values[vi], color=default_color[ki], linewidth=2.5, linestyle="-", label=skeys[ki])
        if (i % group) == 0:
            plt.legend(loc='upper left')
            plt.subplot(221+int(i/group))
    plt.legend(loc='upper left')
    return plt;
