//-----Proyecto Modular-------
//-----Modulo Inalambrico Autopay-------
//-----Ricardo Muñoz Ortiz--------------
//-----Proyecto Modular-------
//-----Modulo Inalambrico Autopay-------
//-----Ricardo Muñoz Ortiz--------------
//-----Jesus Angel Medina Ramirez-------
//-----Ricardo Alejandro Díaz de León Montaño--
//------INCE 2023 A-----------------------------

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QSerialPortInfo>
#include<QSerialPort>
#include<QDebug>
#include<QStringList>
#include<QJsonArray>
#include<QJsonDocument>
#include<QJsonObject>
#include<QJsonValue>
#include<QFile>
#include<QFileDialog>
#include<QMessageBox>
#include<QTextStream>

//LIBREIAS NECESARIAS PARA LA BASE DE DATOS
#include <QSqlDatabase>
#include <QtSql>
#include <QSqlQuery>
#include <QSqlError>


#include <QtNetwork>
#include <QVector>


#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "mat2qimage.h"

//LIBREERIAS NECESARIAS PARA CONECTAR OPENCV CON QT CREATOR
#include<opencv2/core/core.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv/cv.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/video/background_segm.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgcodecs.hpp>

//LIBRERIAS NECESARIAS PARA LA IMPLEMENTACION DE ALGUNAS FUNCIONES DE QT
#include<QTimer>
#include<QDebug>
#include<opencv2/objdetect.hpp>
#include<QDateTime>
#include <QtMultimedia/QMediaPlayer>
#include <Qt>
#include <qdebug.h>
#include<QDebug>
#include <QDirIterator>
#include<QDebug>
#include<QFile>
#include <QTextStream>
#include<QtNetwork/QtNetwork>
#include<QTimer>
#include <QInputDialog>
#include <QDateTime>
#include <QDialog>

//LIBRERIAS PARA CREACIÓN Y EDICIÓN DE ARCHIVOS PDF
#include <QPdfWriter>
#include <QPainter>

#include <QDateTime>


using namespace cv;
VideoCapture camara(0);//Activación de videocamara
Mat imagen;
QString nombreArchivo = "../CaraFrontal.xml"; //archivo HAAR entrenado para detección de caras
CascadeClassifier detector_caras; //Hacer clasificador en cascada para la detección de caras

QPdfWriter pdf("../Recibo.pdf");
QString recibo2 = "";
QString recibofinal = "";


//variables utilizadas en el login

int tiempoEspera =15; //cuanto tiempo tardara en aparecer el dialogo de inicio de sesión
int contador1 = 0 ; //variable de control
bool disparo = true;
int num_caras = 0;

//Variables para checar que la cara se haya detectado por mas de tres segundos

cv::Rect prev_face; // declaración Rect para la cara previa
bool prev_face_detected = false; //declaración booleana para cambiar de valor cuando la cara previa sea la misma a la actual
int num_frames=0; //incrementar el numero de frames segun la cara previa
int64 t0 = cv::getTickCount(); //cuenta el numero de milisegundos que han transcurrido desde que cierta acción empieza
double threshold = 30;
bool encontrada = false;


//----------------------------------MODULO DE INICIO DE SESIÓN POR RECONOCIMIENTO FACIAL ----------------------------
//Declarar vector donde estan guardadas las imagenes para el reconocimiento facial
std::vector<std::string> preloaded_images = {"../Rostro.jpg", "../Rostro2.jpg", "../Rostro3.jpg"};


QSqlDatabase baseDeDatos = QSqlDatabase::addDatabase("QMYSQL"); //Declaración de variable para el control de la base de datos
QString UIDBaseDeDatos;
QString precio;
QString Producto, Peso;
QString CorreoCliente;

int PesoINT, precioINT; // Variables tipo entero para almacenamiento de precio y peso extraido de la base de datos
int diferencia; //Diferencia de peso captada por la celda de carga
int conteo = 0; // Variable de apoyo que incrementa o disminuye conforme los articulos de la lista

//Variables que se emplean para el cambio de función del programa segun se requiera

bool check = false; //Hace el cambio entre agregar o eliminar articulo según se requiera
bool inicio = false; //Hace el cambio entre inicio de sesión y cobro a cliente


int sumatotal = 0;
int i = 0;
int preciovector = 0;
int PosicionLista = 0;
int contadorDif = 0;
bool EnvioCorreo = false;

//Vectores de almacenamiento de datos
QVector<int>ListaPrecios{};//Almacenamiento de todos los precios que se extraen de la base de datos
QVector<int>VectorDif; //Almacenamiento de todos los pesos extraidos de la base de datos para eliminar articulos

MainWindow::MainWindow(QWidget *parent)
  : QMainWindow(parent)
  , ui(new Ui::MainWindow)
{
//Al iniciar el programa se esconden los elementos que no se necesitan para el inicio de sesión
  ui->setupUi(this);
  ui->label_6->hide();
  ui->listWidget->hide();
  ui->lineEdit->hide();
  ui->lineEdit_2->hide();
  ui->lineEdit_4->hide();
  ui->label->hide();
  ui->label_2->hide();
  ui->label_3->hide();
  ui->label_5->hide();
  ui->lcdNumber->hide();
  ui->pushButton->hide();
  ui->pushButton_2->hide();

  conectarArduino();// Llamado a la función para verificar la conexión con Arduino


//Inicialización de la base de datos
  baseDeDatos.setHostName("localhost");
  baseDeDatos.setPort(3306);
  baseDeDatos.setDatabaseName("Lista");
  baseDeDatos.setUserName("admin51");
  baseDeDatos.setPassword("hola1234");

  if(baseDeDatos.open()){
       qDebug() << "Se abrió la base de datos ";
  }
  else{
       qDebug() << "No se abrió la base de datos ";
  }
  QTimer *cronometro = new QTimer(this);
     //Config
     connect(cronometro, SIGNAL(timeout()),this,SLOT(tempo()));
     //incio cronometro
     cronometro->start(50);
     //checar si el detector de caras se abrió correctamente
     if(!detector_caras.load("../HAAR/CaraPerfil.xml")){
         qDebug() << "error al cargar el detector de caras";
     }


}

MainWindow::~MainWindow()
{
  delete ui;
}

void MainWindow::tempo(){
  qDebug() << prev_face_detected; //Mostrar en la terminal si la cara previa es la misma a la actual
  camara >> imagen; //asignar a la variable imagen el frame actual de la camara web/ip


  Mat imagenChica; // variable escalar la imagen
  if(!imagen.empty()){ //condición que se hara efectiva siempre y cuando este activa la camara
      //Cambiar el filtro a gris para la detección de caras
      Mat GRIS; //Declaración para cambiar la imagen a un escala de grises
      cv::resize(imagen, imagenChica, Size(650,350), 0,0,INTER_LINEAR); //cambiar a tamaño 650x350 la imagen actual
      cv::cvtColor(imagenChica, GRIS, COLOR_BGR2GRAY); //Cambiar la imagen actual a una escala de grises


      cv::equalizeHist(GRIS,GRIS); // imagen ecualizada por histogramas
      //declaraciones para la deteccipon de caras
      std::vector<Rect> carasEncontradas; //vector donde se almacenara las caras detectadas
      detector_caras.detectMultiScale( GRIS, carasEncontradas, 1.1, 2, 0|CASCADE_SCALE_IMAGE, // Detección de caras
      Size(30, 30) );
      //imwrite("../IMG0101.jpg",imagenChica); //Tomar una foto durante el programa en caso de que se necesite agregar a la base de datos de las imagenes
      for(const auto& face: carasEncontradas ){  //Ciclo donde a 'face' se le pasara la actual cara encontrada
          qDebug() << prev_face_detected; //mostrar nuevamente si la cara actual es igual a la previa
          cv::rectangle(imagenChica,face,cv::Scalar(0,255,0),2); // dibujar un rectángulo color verde en la cara detectada
          if(prev_face_detected == true){ // condición que se hara efectiva siempre y cuando la cara actual sea igual a la previa
              double dist = norm(face.tl() - prev_face.tl()); //checar la distancia de pixeles entre la cara actual y la previa
              if(dist < 50){ //condición que se hara efectiva siempre y cuando la distancia de pixeles entre ambas imagenes sea menor a 50
                  bool flag = false; //cambiar la bandera a falso
                  for (const auto& preloaded_image : preloaded_images) { //cargar a 'preloaded_image' cada una de las imagenes de la base de datos y trabajar en el ciclo una por uan
                      Mat loaded_image = imread("../" + preloaded_image, COLOR_BGR2GRAY); // cargar la imagen pre cargada y transformarla a una escala de grises


                      std::vector<Rect> faces; //crear un vector nuevo para la detección de caras pero en las imágenes de la base de datos
                      detector_caras.detectMultiScale( loaded_image, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE,
                      Size(30, 30) ); //realizar la detección de caras de las imagenes de la base de datos


                      for (const auto& loaded_face : faces) { //ciclo for para trabajar con cada cara detectada de la base de datos y compararla con la cara actual de la camara
                          double loaded_dist = norm(face.tl() - loaded_face.tl()); //calcular la distancia de los pixeles entre ambas imagenes
                          qDebug() << "DISTANCIA" << loaded_dist; //imprimir en la consola la distancia calculada
                          if (loaded_dist < 30) { //si la distancia es menor a 30/umbral, entrar a la condición
                              qDebug() << "SE ENCONTRO UNA COINCIDENCIA"; // imprimir en la consola que se encontro una coincidencia
                              flag = true; //cambiar la bandera a true
                              encontrada = true; //cambiar la variable a true
                              break;
                          }
                      }
                      if (flag) break;
                  }
                  num_frames++; //incrementar el numero de frames
                  double elapsed_time = (cv::getTickCount() - t0) / cv::getTickFrequency(); // calculo del tiempo elapsado desde que reconocio una cara
                  qDebug() << elapsed_time; //mostrar en la consola el tiempo elapsado


                  if(elapsed_time >5.0 && encontrada == true){ // si reconocio una cara por mas de 5 segundos y esta coincide con alguna cara de la base de datos, entrar a la condición
                      ui->label_3->hide(); //Ocultar camara
                      ui->label->show(); //Mostrar panel para empezar a registrar productos




                  }
                  if (!flag) qDebug() << "NO COINCIDENCIA"; //imprimir en la consola que no se enontro ninguna coincidencia
              }
              else{
                  //reiniciar el numero de frames y el tiempo elpasado en caso de que exista una nueva cara
                  num_frames=0;
                  t0 = cv::getTickCount();
              }
          }
          else{ //cambiar la cara previa a la actual en caso de que se cumpla
              prev_face = face;
              prev_face_detected = true;
          }
          //cambiar la cara previa a la actualizar y evaluar
          prev_face = face;
      }
  }


  QImage imagenQT = Mat2QImage(imagenChica); //asignar la imagen escalada a un label


  //Convertir la imagen QT a un pixmap
  QPixmap mapaPixeles = QPixmap::fromImage(imagenQT);


  //Despejar el contenido de la etiqueta a la que se asignara
  ui->label_6->clear();


  //mostrar el mapa de pixeles en la etiqueta asignada
  ui->label_6->setPixmap(mapaPixeles);
}

void MainWindow::recepcionSerialAsincrona(){ //Esta función se ejecuta cada vez que el puerto serial contiene datos
      //Se verifica que la tarjeta está conectada y es legible
if(tarjeta_conectada && tarjeta->isReadable()){
    //QByteArray datosLeidos = arduino->readAll();
    QByteArray datosLeidos = tarjeta->readLine();
    int indice0 = datosLeidos.indexOf("}");
    int indice1 = datosLeidos.indexOf("{");
    QString infoExtraida = datosLeidos.mid(indice1,(indice0-indice1)+1);

//Realiza la lectura del puerto serie delimitando el inicio y final mediante corchetes
    QJsonDocument mensajeJSON = QJsonDocument::fromJson(infoExtraida.toUtf8());
         QJsonValue varNivel1 = mensajeJSON["UID"];
         QString varNivel1Texto = varNivel1.toString();
         QJsonValue varNivel2 = mensajeJSON["Dif"];
         QString varNivel2Texto = varNivel2.toString();
              qDebug() << "La uid es:" << varNivel1Texto;
         if(!varNivel1Texto.isEmpty()){
//La cadena JSON contiene unicamente dos variables, las cuales se descomponen en las siguientes variables
              ui->lineEdit->setText(varNivel1Texto);
              ui->lineEdit_2->setText(varNivel2Texto);
         }


         if(indice0>=0 && indice1>=0) {
                 qDebug() << "Datos extraidos: " << infoExtraida.toUtf8().constData();


         }
//-------------------------MODO DE ELIMINACIÓN DE PRODUCTO--------------------------
         diferencia = varNivel2Texto.toInt();
         if(diferencia>=20){//Se corrobora que la diferencia captada se considerable para el programa
              check = true;

              for(contadorDif = conteo -1; contadorDif !=-1; contadorDif--){//Se realiza una iteración entre todos los valores del vector de peso

                  int ValorPeso = VectorDif.value(contadorDif);
                  qDebug() <<"Conteo" <<contadorDif << "VEc" <<ValorPeso;
                  qDebug() <<"Diferencia" << diferencia;
                  if(diferencia <= ValorPeso+20 && diferencia >= ValorPeso-20){//Si existe algún valor del vector durante la iteración que sea similar a la diferencia captada se detiene la iteración
                      int itemlist = contadorDif; //Se localiza el elemento de la lista de acuerdo a la posición donde se detuvo la iteración
                      delete ui->listWidget->item(itemlist); //Se elimina dicho elemento de la lista
                      int Resta = ListaPrecios.value(itemlist); //Se localiza el elemento del vector de precios de acuerdo a la posición donde se detuvo la iteración
                      sumatotal = sumatotal-Resta;    //Se descuenta el precio del producto retirado
                      ui->lcdNumber->display(sumatotal);
                      //En ambos vectores se elimina el elemento en cuestión
                      ListaPrecios.remove(itemlist);
                      VectorDif.remove(itemlist);
                      conteo --; //Se decrementa la variable para indicar que un producto se retiró
                      contadorDif = 0;

                  }

               }


              //Envío por el puerto serial la confirmación de la diferencia de peso captada
             if(tarjeta_conectada && tarjeta->isWritable()){
             tarjeta->write("K\n");
             QMessageBox::information(this, tr("ATENCION"), tr("Se detectó una diferencia de peso")); //Mensaje de alerta para el usuario haciendole saber que la celda de carga detectó una diferencia de peso
             }
          else{
                 qDebug () << "No se envió la cadena";
             }
          }
//Al no existir una diferencia de peso considerable para el programa, se hace el cambio al modo "cobro cliente"
        if(inicio == true){
         QString comando1 = "SELECT * FROM ListaArt WHERE UID = "+varNivel1Texto+""; //Se busca en la base datos coincidencias del UID introducido
         QSqlQuery comando2;
         comando2.prepare(comando1);
         if(comando2.exec()){
             while(comando2.next()){
                 //Si existe una coincidencia de UID en la base de datos, se extrae la información de Precio, Peso y nombre del producto
                 UIDBaseDeDatos = comando2.value(3).toString();
                 precio = comando2.value(2).toString();
                 Producto = comando2.value(1).toString();
                 Peso = comando2.value(4).toString();
                  qDebug() << "UID:" << UIDBaseDeDatos << "Producto:" << Producto << "Precio:" << precio ;
                  //La información extraida se convierte a variables de tipo entero
                  precioINT = precio.toInt();
                  PesoINT = Peso.toUInt();
                  tarjeta->clear();

                  //-------------MODO DE ADICIÓN DE PRODUCTOS A LA LISTA---------------------
                  if(check == false){
                  ui->listWidget->show();//Se muestra la lista en la interfaz
                  ui->listWidget->addItem(Producto + "                                                                                      $" + precio);//Se agrega la información extraida de la base de datos en formato especifico a la lista de productos
                  //Se agregan los valores extraidos a sus respectivos vectores
                  ListaPrecios.insert(conteo, precioINT);
                  VectorDif.insert(conteo, PesoINT);
                  qDebug() << ListaPrecios << "Vector peso:" << VectorDif;
                  conteo ++;//Se incrementa la variable para indicar que un producto fue añadido

                      for(i=0; i<1; i++){
                          preciovector = ListaPrecios.value(conteo-1);//Se extrae del vector de precios el valor del ultimo producto agregado
                          qDebug () << preciovector << "Extraido";
                          sumatotal += preciovector;//Se realiza la suma del precio del ultimo producto agregado con la cantidad previamente contenida
                          ui->lcdNumber->display(sumatotal);
                          qDebug () << sumatotal << "Precio";
                      }
                      i = 0;//Retorno de la variable a cero para realizar la suma de precios de demas productos a agregar
                 }
                 else{

                 }
                 check = false;

             }
          if(UIDBaseDeDatos.isEmpty()){
              QMessageBox::information(this, tr("Error"), tr("NO es un producto registrado en la base de datos"));
          }
         }
         UIDBaseDeDatos = "";
}          //--------------MODO INCIO DE SESION----------------------------------
        else{
            ui->lineEdit_3->setText(varNivel1Texto);
            QString NombreCliente;
            QString comando3 = "SELECT * FROM Usuarios WHERE UID = "+varNivel1Texto+""; //Busca en la base de datos algún usuario que corresponda con el UID ingresado
            QSqlQuery comando4;
            comando4.prepare(comando3);
              if(comando4.exec()){ //Si existe un usuario con ese UID se muestran los demas elementos de la interfaz y se habilita el modo cobro cliente
                  while(comando4.next()){
                     NombreCliente = comando4.value(1).toString();
                     CorreoCliente = comando4.value(3).toString();
                     ui->lineEdit->show();
                     ui->lineEdit_2->show();
                     ui->lineEdit_4->show();
                     ui->label->show();
                     ui->label_2->show();
                     ui->label_3->show();
                     ui->label_5->show();
                     ui->lcdNumber->show();
                     ui->pushButton->show();
                     ui->pushButton_2->show();
                     ui->lineEdit_3->hide();
                     ui->lineEdit_3->clear();
                     ui->label_4->hide();
                     ui->lineEdit_4->setText(NombreCliente);
                     inicio = true;

                  }
                  if(NombreCliente.isEmpty()){
                      //En caso de no existir un cliente con ese UID se alerta la interfaz notificando del problema
                      QMessageBox::information(this, tr("Error"), tr("NO es un cliente registrado en la base de datos"));
                      inicio = false;
                      ui->lineEdit_4->clear();
                  }
              }
        }

}

}


void MainWindow::conectarArduino(){
//Declaracion de las variables
tarjeta = new QSerialPort(this);
connect(tarjeta, &QSerialPort::readyRead, this, &MainWindow::recepcionSerialAsincrona);
QString nombreDispositivoSerial = "";
int productoID = 0;
int fabricanteID = 0;

//Imprimir en la terminal, la cantidad de dispositivos seriales virtuales, conectados a la computadora
qDebug() << "Esto es un mensaje de terminal" << endl;
qDebug() << "Puertos disponibles: " << QSerialPortInfo::availablePorts().length();

foreach (const QSerialPortInfo &serialPortInfo, QSerialPortInfo::availablePorts()){
    if(serialPortInfo.hasVendorIdentifier()){
        fabricanteID = serialPortInfo.vendorIdentifier();
        productoID = serialPortInfo.productIdentifier();
        nombreDispositivoSerial = serialPortInfo.portName();

        qDebug() << "Fabricante: " << fabricanteID << endl;
        qDebug() << "Producto:   " << productoID << endl;

    }
}


//Conexion con Arduino
if(productoID == ArduinoUNO || productoID == ArduinoMEGA){
    //Establece TTYACMX como nombre del dispositivo
    tarjeta->setPortName(nombreDispositivoSerial);
    tarjeta->open(QIODevice::ReadWrite);
    tarjeta->setDataBits(QSerialPort::Data8);
    tarjeta->setBaudRate(QSerialPort::Baud115200);
    tarjeta->setParity(QSerialPort::NoParity);
    tarjeta->setStopBits(QSerialPort::OneStop);
    tarjeta->setFlowControl(QSerialPort::NoFlowControl);
    tarjeta_conectada = true;
}
}


void MainWindow::on_pushButton_clicked(){ //Esta función de ejecuta cada que se oprime el botón "Finalizar"
  int longitud, y = 2500;
   QPainter painter(&pdf); //Creación del documento PDF

   QImage carroRFID ("../CarroRFID.jpeg"); //Toma la imagen de la ruta especificada
   QImage carrofinal;
   carrofinal.load("../carropng.png");
   QPixmap MapCarro = QPixmap::fromImage(carroRFID); //Convierte dicha imagen a mapa de pixeles
   QPixmap mapcarrofinal = QPixmap::fromImage(carrofinal);
   QPixmap MapCarroResize = MapCarro.scaled(1500,1600); //Se reescala la imagen de acuerdo al tamaño deseado
   QPixmap MapCarroFinalResize = mapcarrofinal.scaled(800, 800);

   QDateTime horafecha = QDateTime::currentDateTime();
   QString Fecha = horafecha.toString("dd/MM/yyyy HH:mm");

   //Personalización del formato de diseño del ticket de compra
   QFont fontTitulo("Helvetica", 30);
   QFont fontSubtitulo("Helvetica", 16);
   QFont fontrecibo("Futura", 14);
   QFont fontFinal("Gotham", 16);
   painter.setFont(fontTitulo);
   painter.drawText(3000, 500, "SUPER CUCEI");
   painter.setFont(fontSubtitulo);
   painter.drawText(2000, 800,"Blvd. Gral. Marcelino García Barragán 1421");
   painter.drawText(2200, 1100,  "Olímpica, 44430 Guadalajara, Jal.");
   painter.drawText(3300, 1400, "Tel: 33 1378 5900");
   painter.drawText(300, 1800, "***************************************************************************************");
   painter.drawText(1000, 2200, "Productos");
   painter.drawText(8200, 2200, "Precio");
   painter.drawPixmap(500,1, MapCarroResize);

for (longitud = 0; longitud < ui->listWidget->count(); longitud++){ //Se realiza una iteración entre todos los productos listados
  QListWidgetItem *hora = ui->listWidget->item(longitud); //Se extrae cada elemento encontrado en la iteración
  QString recibofinal = hora->text(); //Pasandolo a una variable QString
  painter.setFont(fontrecibo);
  painter.drawText(1000, y, recibofinal); //Imprime en el documento PDF la variable QString anterior en una posición movil regida por la variable "y"
  y += 350; //Se incrementa la variable para imprimir el siguiente elemento en una posición diferente a la anterior

}
int final = ui->lcdNumber->value();
QString Stringfinal = QString::number(final);
QString conteofinal = QString::number(conteo);
painter.setFont(fontFinal);
painter.drawImage(QRect(1000, y-350, 1000, 1000), carrofinal);
painter.drawText(1500, y+150, conteofinal);
painter.drawText(8200, y+200, "$" + Stringfinal);
painter.drawText(1500, 12000,"La compra fue realizada el día : " + Fecha);

painter.end(); //Finalización del control del documento PDF

//------------RESET DE TODAS LAS VARIABLES EMPLEADAS PARA UN NUEVO COBRO A CLIENTE--------------------


inicio = false;
sumatotal = 0;
ListaPrecios.clear();
VectorDif.clear();
UIDBaseDeDatos = "";
Producto = "";
precio = "";
conteo = 0;

y = 0;

ui->lineEdit_3->show();
ui->label_4->show();
ui->lineEdit->hide();
ui->lineEdit_2->hide();
ui->lineEdit_4->hide();
ui->label->hide();
ui->label_2->hide();
ui->label_3->hide();
ui->label_5->hide();
ui->lcdNumber->hide();
ui->pushButton->hide();
ui->pushButton_2->hide();
ui->listWidget->clear();
ui->listWidget->hide();
ui->lcdNumber->display(0);

tempo();
disparo =true;
ui->label_6->show();

//--------------EJECUCIÓN DE COMANDOS PARA HABILITAR TERMINAL Y ENVIAR CORREO CON TICKET PDF ADJUNTO-------------------------------
QString Asunto = "Recibo de compra" ;
QString direccion = "uuencode /home/ricardo/Recibo.pdf Recibo.pdf | mail -s \""+Asunto+"\" "+CorreoCliente+"";
std::string comando = direccion.toStdString();
system(comando.c_str());
CorreoCliente = "";
}


void MainWindow::on_pushButton_2_clicked(){ //Esta función se ejecuta cada vez que se presiona el botón "Eliminar Articulo"

   PosicionLista = ui->listWidget->currentRow(); //Obtener la fila del elemento seleccionado
   delete ui->listWidget->item(PosicionLista); //Eliminación del elemento en la lista
   int ProductoEliminar = ListaPrecios.value(PosicionLista); //Precio correspondiente en el vector de precios
   sumatotal = sumatotal - ProductoEliminar; //Se descuenta el precio del producto retirado
   ui->lcdNumber->display(sumatotal);
   //En ambos vectores se elimina el elemento en cuestión
   ListaPrecios.remove(PosicionLista);
   VectorDif.remove(PosicionLista);
   conteo --; //Se decrementa la variable para indicar que un producto se retiró
   qDebug() << ListaPrecios;
}
