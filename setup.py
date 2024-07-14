from setuptools import setup,find_packages

setup(
    name='moiregenerate',
    version='0.1',
    author="zhao luneng",
    author_email="1713175349@qq.com",
    description="generate moire pattern",
    license="MIT",
    keywords="moire pattern",
    package=["moiregenrate"],
    scripts=["bin/moiregenerate-cmd","bin/moiregenerate-cmd-nep"],
    install_requires=["numpy","matplotlib","ase"],
)