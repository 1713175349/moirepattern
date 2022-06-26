from setuptools import setup,find_packages

setup(
    name='moiregemerate',
    version='0.1',
    author="zhao luneng",
    author_email="1713175349@qq.com",
    description="generate moire pattern",
    license="MIT",
    keywords="moire pattern",
    package=["moiregemerate"],
    scripts=["bin/moiregenerate-cmd"],
    install_requires=["numpy","matplotlib","ase"],
)