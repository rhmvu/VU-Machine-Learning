import svm, lin_reg, basic_lin_reg

print('Basic Linear Regression Model\n\t Variance score: {}\n'.format(basic_lin_reg.run()))
print('Advanced Linear Regression Model\n\t Variance score: {}\n'.format(lin_reg.run()))
print('Support Vector Machine (regression)\n\t Variance score: {}'.format(svm.run()))