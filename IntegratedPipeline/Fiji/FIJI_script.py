from ij import IJ, WindowManager
import os

for tipo in ["positive", "negative", "condensed", "multiple"]:
    print("Tipo:", tipo)
    # === Percorsi ===
    input_dir = "/home/francesco/Scaricati/Dataset/Images/test/{}/".format(tipo)
    output_kill_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/DataSet_test/GrayImages/{}/".format(tipo)
    output_mask_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/DataSet_test/MaskedImages/{}/".format(tipo)
    
    # Crea le cartelle di output se non esistono
    for dir_path in [output_kill_dir, output_mask_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Creata cartella:", dir_path)

    # Filtra i file .tif nella cartella di input
    file_list = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
    
    for file_name in file_list:
        input_path = os.path.join(input_dir, file_name)
        output_kill_path = os.path.join(output_kill_dir, file_name)
        output_mask_path = os.path.join(output_mask_dir, file_name)

        print("Elaborazione:", file_name)

        # Apri immagine
        imp = IJ.openImage(input_path)
        if imp is None:
            print("Errore: impossibile aprire", file_name)
            continue

        # Applica Kill Borders
        IJ.run(imp, "Kill Borders", "")
        imp_kill = WindowManager.getCurrentImage()

        IJ.saveAs(imp_kill, "Tiff", output_kill_path)
        print("Salvato (Kill Borders):", output_kill_path)
        
        # Converti in 8-bit prima della sogliatura
        IJ.run(imp_kill, "8-bit", "")
        IJ.setRawThreshold(imp_kill, 15, 255)
        IJ.run(imp_kill, "Convert to Mask", "")
        IJ.saveAs(imp_kill, "Tiff", output_mask_path)
        print("Salvato (Maschera):", output_mask_path)

        # Chiudi tutte le immagini aperte
        while WindowManager.getImageCount() > 0:
            img = WindowManager.getCurrentImage()
            if img is not None:
                img.close()
