import math

def pearson(data_compare):
	x = [i[0] for i in data_compare]
	y = [i[1] for i in data_compare]
	n = len(data_compare)
	mu_x = average(x)
	mu_y = average(y)
	sigma_x = std(x, mu_x)
	sigma_y = std(y, mu_y)
	
	num = sum([(i[0] * i[1]) for i in data_compare]) - n * mu_x * mu_y
	det = (n - 1) * sigma_x * sigma_y
	
	return float(num)/det
	
def average(x):
	return sum(x)/float(len(x))
	
def std(x, mu):
	return (sum([(i - mu)**2 for i in x]) / float(len(x)))**0.5
