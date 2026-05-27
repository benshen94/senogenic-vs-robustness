// Assemble Figure 4 as fully editable Illustrator vector artwork.
// This script opens each source PDF, copies its page objects into the target
// document, and saves an editable AI/PDF. It does not place PDF or PNG panels.

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
var topToMiddleExtraGap = 70;
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

function duplicatePdfArtworkInto(doc, spec, left, top, name) {
    var sourceFile = new File(spec.path);
    if (!sourceFile.exists) {
        throw new Error("Missing source PDF: " + sourceFile.fsName);
    }

    var sourceDoc = app.open(sourceFile);
    app.activeDocument = sourceDoc;
    sourceDoc.selectObjectsOnActiveArtboard();
    app.copy();
    sourceDoc.close(SaveOptions.DONOTSAVECHANGES);

    app.activeDocument = doc;
    var targetLayer = doc.activeLayer;
    app.paste();

    var pastedItems = [];
    for (var i = 0; i < doc.selection.length; i++) {
        pastedItems.push(doc.selection[i]);
    }
    if (pastedItems.length === 0) {
        throw new Error("No editable artwork pasted from: " + sourceFile.fsName);
    }

    var group = targetLayer.groupItems.add();
    group.name = name;
    for (var j = pastedItems.length - 1; j >= 0; j--) {
        pastedItems[j].move(group, ElementPlacement.PLACEATBEGINNING);
    }

    var bounds = group.visibleBounds;
    var currentLeft = bounds[0];
    var currentTop = bounds[1];
    group.translate(left - currentLeft, top - currentTop);
    return group;
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
duplicatePdfArtworkInto(doc, panels.ab, abLeft, topY, "fig4_ab_sweden_period_projection");

var middleY = topY - panels.ab.height - rowGap - topToMiddleExtraGap;
var middleLeft = (artWidth - middleWidth) / 2;
duplicatePdfArtworkInto(doc, panels.c, middleLeft, middleY, "Fig4C");
duplicatePdfArtworkInto(doc, panels.d, middleLeft + panels.c.width + middleGap, middleY, "Fig4D_extrap");

var bottomY = middleY - panels.c.height - rowGap;
var bottomLeft = (artWidth - panels.e.width) / 2;
duplicatePdfArtworkInto(doc, panels.e, bottomLeft, bottomY, "sweden_sr_contour_projection");
addPanelLabel(doc, "e", bottomLeft - 42, bottomY + 12);

// The imported PDF page boxes contain extra invisible whitespace above the top
// row. Crop the artboard around the visible composition without moving artwork.
doc.artboards[0].artboardRect = [0, 1705, artWidth, -46];

var aiFile = new File(outputDir + "/Figure4_editable.ai");
var aiOptions = new IllustratorSaveOptions();
aiOptions.pdfCompatible = true;
doc.saveAs(aiFile, aiOptions);

var pdfFile = new File(outputDir + "/Figure4_editable.pdf");
var pdfOptions = new PDFSaveOptions();
pdfOptions.preserveEditability = true;
doc.saveAs(pdfFile, pdfOptions);

var pngFile = new File(outputDir + "/Figure4_editable.png");
var pngOptions = new ExportOptionsPNG24();
pngOptions.antiAliasing = true;
pngOptions.artBoardClipping = true;
pngOptions.transparency = false;
pngOptions.horizontalScale = 416.6667;
pngOptions.verticalScale = 416.6667;
doc.exportFile(pngFile, ExportType.PNG24, pngOptions);

doc.saveAs(aiFile, aiOptions);
