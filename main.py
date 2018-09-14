import numpy as np

from calcACC import calcACC
from hungarian import Hungarian

def main():
	groundTruth = np.array([1,1,1,1,1,  2,2,2,2,2,  3,3,3,3,3])
	predValue =   np.array([2,2,2,2,2,  3,3,3,3,3,  1,1,1,1,1])
	acc = calcACC(groundTruth, predValue)
	print(acc)


if __name__ == '__main__':
	main()