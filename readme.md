### How to get the SLProject

The SLProject is hosted at GitHub as a GIT repository.
[GIT](http://git-scm.com/) is a distributed versioning control system.

**Please do not download the ZIP-file!** You won't be able to build the project because the zip file misses the submodules.

The SLProject contains some GIT submodules. 
To clone all at once you need to clone recursively.
You can clone recursively either with the GitHub GUI-Tool or with the following command in the console:

```
git clone --recursive https://github.com/cpvrlab/SLProject.git
cd SLProject
git submodule update --init --recursive
```

For the latest development please checkout the develop branch:

```
git checkout develop
```

For detailed build instructions on various platforms go to the [SLProject wiki](https://github.com/cpvrlab/SLProject/wiki).

[![Build Status](https://ci.appveyor.com/api/projects/status/d101mkgdfy4lqe01?svg=true)](https://ci.appveyor.com/project/MarcusHudritsch/slproject)

See the [SLProject Homepage](http://cpvrlab.github.io/SLProject_doc/) for more information.
Or go directly to the html documentation here: [framework documentation](http://cpvrlab.github.io/SLProject_doc/html/index.html)
