Relative Pose Estimation using Two Correspondences
==================================================

Code for our ECCV 2018 paper:
[Affine Correspondences between Central Cameras for Rapid Relative Pose Estimation](http://openaccess.thecvf.com/content_ECCV_2018/html/Ivan_Eichhardt_Affine_Correspondences_between_ECCV_2018_paper.html)
Ivan Eichhardt, Dmitry Chetverikov; The European Conference on Computer Vision (ECCV), 2018, pp. 482-497

Cite it as
```
@InProceedings{Eichhardt_2018_ECCV,
	author = {Eichhardt, Ivan and Chetverikov, Dmitry},
	title = {Affine Correspondences between Central Cameras for Rapid Relative Pose Estimation},
	booktitle = {The European Conference on Computer Vision (ECCV)},
	month = {September},
	year = {2018}
}
```

Build
-----

...

Third-party libraries
---------------------

Included with the repository:
- Eigen
- [TheiaSfM](https://github.com/sweeneychris/TheiaSfM) (Modified, added [LO+](http://cmp.felk.cvut.cz/software/LO-RANSAC/Lebeda-2012-Fixing_LORANSAC-BMVC.pdf))
- glog
- gflags

Used as inspiration:
- [Google Ceres Solver](http://ceres-solver.org/)
- [OpenMVG](https://github.com/openMVG/openMVG)

TODOs
-----

- Add more camera models
- Add more inputs
- Provide visual output
- Provide tools for feature extraction
- Clean up automatic differentiation module
- Use Cmake for maintainability
- ...
