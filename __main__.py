import lin_reg_L2
import lin_reg_L1
import basic_lin_reg
import KNN
from multiprocessing import Process
from result import Result

model = {
    "Basic Linear Regression Model": basic_lin_reg,
    "L1 regularized Regression Model": lin_reg_L1,
    "L2 regularized Regression Model": lin_reg_L2,
    "K Nearest Neighbours": KNN
}

def run_model(model_name, model):
    result = model.run()
    print('{}\n\
        \tVariance score: {}\n\
        \tMean Squared Error: {}\n\
        \tMedian Absolute Error: {}\n'.format(model_name, result.score(), result.mse(), result.mae()))


if __name__ == '__main__':
    for name in model:
        p = Process(target=run_model, args=(name, model[name]))
        p.start()