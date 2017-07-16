# clear cache
rm -rf build/*
rm -rf dist/*

# build universal wheel
python setup.py bdist_wheel

# upload to PyPi
twine upload dist/*
