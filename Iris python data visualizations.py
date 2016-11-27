# Taken from the tutorial found here:
# https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook

# First we'll import pandas, a data processing and CSV file I/O library
import pandas as pd

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll want ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

# Next, we'll load the Iris flower dataset, which is in the "../datasets/" directory
iris = pd.read_csv("./datasets/Iris.csv") # the iris dataset is now a Pandas Dataframe

# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do
iris.head()
##plt.show()

iris["Species"].value_counts()

# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Iris features.
iris.plot(kind="scatter", x="Sepal.Length", y="Sepal.Width")

##plt.show()

# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="Sepal.Length", y="Sepal.Width", data=iris, size=5)
##plt.show()

# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(iris, hue="Species", size=5) \
                    .map(plt.scatter, "Sepal.Length", "Sepal.Width") \
                    .add_legend()
##plt.show()

# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="Petal.Length", data=iris)
##plt.show()

# One way we can extend this plot by adding a later of individual points on top of
# it through Seaborn's striplot
#
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Species", y="Petal.Length", data=iris)
ax = sns.stripplot(x="Species", y="Petal.Length", data=iris, jitter=True, edgecolor="gray")
##plt.show()

# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Species", y="Petal.Length", data=iris, size=6)
##plt.show()

# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernal density estimate of the underlying feature
sns.FacetGrid(iris, hue="Species", size=6) \
                    .map(sns.kdeplot, "Petal.Length") \
                    .add_legend()
##plt.show()

# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
#
# From the pairplot, we'll see that the Iris-setosa species is separated from the other
# two across all feature combinations
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
##plt.show()

# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as kde
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
##plt.show()

# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
##plt.show()

# One cool more sophisticated technique pandas has available is called Andrew Curves
# Andrew Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
##from pandas.tools.plotting import andrews_curves
##andrews_curves(iris.drop("Id", axis=1), "Species")
##plt.show()

# Another multivariate visualization technique pandas has is parallel_coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")
##plt.show()

# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and them simulates
# having each sample attached to those points throug a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")
plt.show()
