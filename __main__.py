from multiprocessing import Process
import sys
import basic_lin_reg
import KNN
import lin_reg_L1
import lin_reg_L2
from prettytable import PrettyTable

table = PrettyTable(
    ["Model", "Variance Score", "Mean Squared Error", "Median Absolute Error", "R2 Score"])

model = {
    "Basic Linear Regression Model": basic_lin_reg,
    "L1 Regularized Regression Model": lin_reg_L1,
    "L2 Regularized Regression Model": lin_reg_L2,
    #"K Nearest Neighbours (gridsearch)": KNN
}


def run_model(model_name, model):
    result = model.run()

    # Round all floats
    for i in range(0, len(result)):
        result[i] = float(round(result[i], 6))

    table.add_row([model_name] + result)
    # print('{}\n\
    #     \tVariance score: {}\n\
    #     \tMean Squared Error: {}\n\
    #     \tMedian Absolute Error: {}\n\
    #     \tR2 Score: {}'.format(model_name, round(result[0], 3), round(result[1], 3), round(result[2], 3),round(result[3], 3)))


if __name__ == '__main__':
    # p = None
    for name in model:
        print("\rrunning model: ", name, end='')
        sys.stdout.flush()
        # p = Process(target=run_model, args=(name, model[name]))
        # p.start()
        run_model(name, model[name])
    # p.join()
    print("\r", end='')
    print(table)
