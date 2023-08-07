import setuptools

setuptools.setup(
    name="yolov4_attendance",
    version='2.0.0',
    description="Script de classification d'image basé sur le modèle yolov4",
    maintainer="Parc national des Écrins",
    url="https://github.com/PnEcrins/yolov4-attendance",
    packages=setuptools.find_packages(where="."),
    package_data={
        'model': ['model/yolov4.h5']
    },
    install_requires=(list(open("requirements.txt","r"))),
    python_requires='>=3.9'
)