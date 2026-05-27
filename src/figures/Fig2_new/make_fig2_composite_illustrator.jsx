// Compose the new Fig. 2 from vector panel PDFs in Adobe Illustrator.
// Run from Illustrator ExtendScript. Outputs AI, PDF, and PNG preview.

#target illustrator

(function () {
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    var root = "/Users/benshenhar/Library/CloudStorage/GoogleDrive-benshenhar@gmail.com/My Drive/Weizmann/Alon Lab/Aging/python/notebooks/thresholds, noise";
    var outputDir = root + "/Figures_new/Fig2_new";

    var paths = {
        a: outputDir + "/fig2a_new.pdf",
        b: outputDir + "/fig2b_new.pdf",
        c: outputDir + "/fig2c_new.pdf",
        de: outputDir + "/fig2de_new.pdf",
        ai: outputDir + "/Fig2_new.ai",
        pdf: outputDir + "/Fig2_new.pdf",
        png: outputDir + "/Fig2_new.png"
    };

    var artboardWidth = 1805.82;
    var artboardHeight = 1520.33;
    var finalBottomCrop = 60;

    var doc = app.documents.add(DocumentColorSpace.RGB, artboardWidth, artboardHeight);
    doc.artboards[0].artboardRect = [0, artboardHeight, artboardWidth, finalBottomCrop];

    var topGroupWidth = 1780;
    var topGap = 35;
    var topPanelWidth = (topGroupWidth - 2 * topGap) / 3;
    var topPanelHeight = topPanelWidth * (352.8 / 482.4);
    var topX = (artboardWidth - topGroupWidth) / 2;
    var topY = artboardHeight - 80;

    var bottomWidth = 1780;
    var bottomHeight = bottomWidth * (629.61835 / 1259.2680854072);
    var bottomX = (artboardWidth - bottomWidth) / 2;
    var bottomY = topY - topPanelHeight - 40;

    placePdf(doc, paths.b, topX, topY, topPanelWidth, topPanelHeight);
    placePdf(doc, paths.a, topX + topPanelWidth + topGap, topY, topPanelWidth, topPanelHeight);
    placePdf(doc, paths.c, topX + 2 * (topPanelWidth + topGap), topY, topPanelWidth, topPanelHeight);
    placePdf(doc, paths.de, bottomX, bottomY, bottomWidth, bottomHeight);

    addPanelLabel(doc, "a", 12, topY + 25);
    addPanelLabel(doc, "b", topX + topPanelWidth + topGap - 42, topY + 25);
    addPanelLabel(doc, "c", topX + 2 * (topPanelWidth + topGap) - 42, topY + 25);

    saveAi(doc, paths.ai);
    savePdf(doc, paths.pdf);
    exportPng(doc, paths.png);

    function placePdf(documentRef, filePath, left, top, width, height) {
        var placed = documentRef.placedItems.add();
        placed.file = new File(filePath);
        placed.position = [left, top];
        placed.width = width;
        placed.height = height;
        placed.embed();
        return placed;
    }

    function addPanelLabel(documentRef, label, left, top) {
        var text = documentRef.textFrames.add();
        text.contents = label;
        text.position = [left, top];
        text.textRange.size = 64;
        text.textRange.characterAttributes.textFont = app.textFonts.getByName("ArialMT");
        text.textRange.characterAttributes.fillColor = rgb(0, 0, 0);
        return text;
    }

    function rgb(red, green, blue) {
        var color = new RGBColor();
        color.red = red;
        color.green = green;
        color.blue = blue;
        return color;
    }

    function saveAi(documentRef, filePath) {
        var options = new IllustratorSaveOptions();
        options.pdfCompatible = true;
        options.embedICCProfile = false;
        documentRef.saveAs(new File(filePath), options);
    }

    function savePdf(documentRef, filePath) {
        var options = new PDFSaveOptions();
        options.preserveEditability = true;
        options.compatibility = PDFCompatibility.ACROBAT8;
        options.generateThumbnails = true;
        options.optimization = true;
        documentRef.saveAs(new File(filePath), options);
    }

    function exportPng(documentRef, filePath) {
        var options = new ExportOptionsPNG24();
        options.artBoardClipping = true;
        options.antiAliasing = true;
        options.transparency = false;
        options.horizontalScale = 200;
        options.verticalScale = 200;
        documentRef.exportFile(new File(filePath), ExportType.PNG24, options);
    }
})();
