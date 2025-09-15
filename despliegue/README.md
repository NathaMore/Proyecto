# Acá se encuentra la guía para el despliegue en AWS EC2

# 1. Crear la instancia
ingrese a AWS -> EC2 -> -> Lance una instancia asignando un nombre adeacuado, 
seleccione imagen Amazon Linux,
tipo de instancia: t2.micro,
cree un nuevo par de claves .pem creada en un localmente en un sitio de facil acceso,
por ultimo, se clic en lanzar instancia
luego asigné un grupo de seguridad que permita el puerto 8050

# 2. Conectarse a la instancia
en una terminal escriba ssh -i llave.pem ec2-user@<IP_PUBLICA>

# 3. Configuración de la instancia y lanzamiento del tablero
verifique actualizaciones del sistema,
en otra terminal escriba scp -i llave.pem dashboard..py ec2-user@IP:/home/ec2-user

# 4. lanzamiento del tablero en el servidor
En la maquina virtual corra la aplicación con python3 dashboard..py

