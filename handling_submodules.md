### How to handle submodules

##### Adding data

1. Add new files to the submodules folder or change existing files.
2. Go into the submodules folder.
3. The submodule will be in a detached head state, check out the desired branch (ex. master).
4. Commit your changes as you normally would.
5. Go back to the root directory and commit the changes made to the submodule.


In short:
```
cd submodule_folder
git checkout master
git add <files>
git commit -m "changes"
cd ..
git add submodule_folder
git commit -m "updated submodule_folder"
```