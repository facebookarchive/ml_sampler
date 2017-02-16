from distutils.core import setup

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.
files = []

setup(name = "ml_sampler",
    version = "0.21",
    description = "",
    author = "Spencer Beecher",
    author_email = "spencebeecher@gmail.com",
    #url = "",
    #Name the folder where your packages live:
    #(If you have other packages (dirs) or modules (py files) then
    #put them into the package directory - they will be found
    #recursively.)
    packages = ['ml_sampler'],
    #'package' package must contain files (see list above)
    #I called the package 'package' thus cleverly confusing the whole issue...
    #This dict maps the package name =to=> directories
    #It says, package *needs* these files.
    #package_data = {},
    #'runner' is in the root.
    #scripts = [],
    long_description = """Take a model assisted sample"""
    #
    #This next part it for the Cheese Shop, look a little down the page.
    #classifiers = []
)
