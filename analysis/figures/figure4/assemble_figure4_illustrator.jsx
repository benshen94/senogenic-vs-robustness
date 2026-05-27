// Assemble the new Figure 4 composite in Adobe Illustrator.
// Source panels are placed at native PDF size to preserve their proportions.

var outputDir = File($.fileName).parent.parent.parent.parent.fsName + "/Figures/Figure4";

var panels = {
    ab: {
        path: outputDir + "/fig4_ab_sweden_period_projection.pdf",
        width: 950.25,
        height: 695.32
    },
    c: {
        path: outputDir + "/Fig4C.pdf",
        width: 477.37,
        height: 364.33
    },
    d: {
        path: outputDir + "/Fig4D_extrap.pdf",
        width: 532.02,
        height: 364.33
    },
    e: {
        path: outputDir + "/sweden_sr_contour_projection_1900_2100_n1m.pdf",
        width: 936.00,
        height: 547.20
    }
};

var margin = 54;
var rowGap = 52;
var middleGap = 44;
var middleWidth = panels.c.width + middleGap + panels.d.width;
var contentWidth = Math.max(panels.ab.width, middleWidth, panels.e.width);
var artWidth = contentWidth + 2 * margin;
var artHeight = margin + panels.ab.height + rowGap + panels.c.height + rowGap + panels.e.height + margin;

function rgbColor(red, green, blue) {
    var color = new RGBColor();
    color.red = red;
    color.green = green;
    color.blue = blue;
    return color;
}

function placePdf(doc, spec, left, top, name) {
    var item = doc.placedItems.add();
    item.file = new File(spec.path);
    item.name = name;
    item.position = [left, top];
    return item;
}

function addPanelLabel(doc, label, left, top) {
    var text = doc.textFrames.pointText([left, top]);
    text.contents = label;
    text.textRange.characterAttributes.size = 28;
    text.textRange.characterAttributes.fillColor = rgbColor(0, 0, 0);
    try {
        text.textRange.characterAttributes.textFont = app.textFonts.getByName("ArialMT");
    } catch (error) {
        text.textRange.characterAttributes.textFont = app.textFonts.getByName("Helvetica");
    }
    return text;
}

var doc = app.documents.add(DocumentColorSpace.RGB, artWidth, artHeight);
doc.artboards[0].artboardRect = [0, artHeight, artWidth, 0];
doc.rulerUnits = RulerUnits.Points;

var topY = artHeight - margin;
var abLeft = (artWidth - panels.ab.width) / 2;
placePdf(doc, panels.ab, abLeft, topY, "fig4_ab_sweden_period_projection");

var middleY = topY - panels.ab.height - rowGap;
var middleLeft = (artWidth - middleWidth) / 2;
placePdf(doc, panels.c, middleLeft, middleY, "Fig4C");
placePdf(doc, panels.d, middleLeft + panels.c.width + middleGap, middleY, "Fig4D_extrap");

var bottomY = middleY - panels.c.height - rowGap;
var bottomLeft = (artWidth - panels.e.width) / 2;
placePdf(doc, panels.e, bottomLeft, bottomY, "sweden_sr_contour_projection");
addPanelLabel(doc, "e", bottomLeft - 42, bottomY + 12);

var aiFile = new File(outputDir + "/Figure4.ai");
var aiOptions = new IllustratorSaveOptions();
aiOptions.pdfCompatible = true;
doc.saveAs(aiFile, aiOptions);

var pdfFile = new File(outputDir + "/Figure4.pdf");
var pdfOptions = new PDFSaveOptions();
pdfOptions.preserveEditability = true;
doc.saveAs(pdfFile, pdfOptions);

var pngFile = new File(outputDir + "/Figure4.png");
var pngOptions = new ExportOptionsPNG24();
pngOptions.antiAliasing = true;
pngOptions.artBoardClipping = true;
pngOptions.transparency = false;
pngOptions.horizontalScale = 416.6667;
pngOptions.verticalScale = 416.6667;
doc.exportFile(pngFile, ExportType.PNG24, pngOptions);

doc.saveAs(aiFile, aiOptions);
