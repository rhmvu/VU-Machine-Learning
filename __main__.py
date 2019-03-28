import lin_reg_L2
import basic_lin_reg
import KNN
from multiprocessing import Process

# model = {
#     "Basic Linear Regression Model": basic_lin_reg,
#     "Advanced Linear Regression Model": lin_reg,
#     "Support Vector Machine (regression)": svm,
#     "K Nearest Neighbours": KNN
# }

model = {
    "ridge": lin_reg_L2
}

def run_model(model_name, model):
    print('{}\n\t Variance score: {}\n'.format(model_name, model.run().score()))


if __name__ == '__main__':
    for name in model:
        p = Process(target=run_model, args=(name, model[name]))
        p.start()