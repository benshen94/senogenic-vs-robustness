// Replace top-row placed PDF panels in Fig2_new.pdf with editable vector/text objects.
// This preserves the current composite document while removing opaque PlacedItem wrappers.

#target illustrator

(function () {
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    var root = "/Users/benshenhar/Library/CloudStorage/GoogleDrive-benshenhar@gmail.com/My Drive/Weizmann/Alon Lab/Aging/python/notebooks/thresholds, noise";
    var outputDir = root + "/Figures_new/Fig2_new";
    var targetPath = outputDir + "/Fig2_new.pdf";
    var panels = [
        outputDir + "/fig2b_new.pdf",
        outputDir + "/fig2a_new.pdf",
        outputDir + "/fig2c_new.pdf"
    ];

    var artboardWidth = 1805.82;
    var artboardHeight = 1520.33;
    var topGroupWidth = 1780;
    var topGap = 35;
    var topPanelWidth = (topGroupWidth - 2 * topGap) / 3;
    var topPanelHeight = topPanelWidth * (352.8 / 482.4);
    var topX = (artboardWidth - topGroupWidth) / 2;
    var topY = artboardHeight - 80;
    var bleed = 8;

    var target = openOrFindDocument(targetPath);
    app.activeDocument = target;

    removeTopRowArtifacts(target);

    for (var i = 0; i < panels.length; i++) {
        var left = topX + i * (topPanelWidth + topGap);
        pastePanelAsEditable(target, panels[i], left, topY, topPanelWidth, topPanelHeight);
    }
    applyPastedArtboardCorrection(target);
    replaceTopRowTitles(target);

    savePdf(target, targetPath);

    function openOrFindDocument(path) {
        for (var i = 0; i < app.documents.length; i++) {
            try {
                if (app.documents[i].fullName.fsName === path) {
                    return app.documents[i];
                }
            } catch (error) {}
        }
        return app.open(new File(path));
    }

    function removeTopRowArtifacts(documentRef) {
        var panelLeft = topX - bleed;
        var panelRight = topX + topGroupWidth + bleed;
        var panelTop = topY + bleed;
        var panelBottom = topY - topPanelHeight - bleed;

        for (var i = documentRef.placedItems.length - 1; i >= 0; i--) {
            var item = documentRef.placedItems[i];
            var bounds = item.visibleBounds;
            var centerX = (bounds[0] + bounds[2]) / 2;
            var centerY = (bounds[1] + bounds[3]) / 2;
            if (
                centerX >= panelLeft &&
                centerX <= panelRight &&
                centerY <= panelTop &&
                centerY >= panelBottom
            ) {
                item.remove();
            }
        }

        for (var j = documentRef.groupItems.length - 1; j >= 0; j--) {
            var group = documentRef.groupItems[j];
            if (String(group.name).indexOf("editable_") !== 0) {
                continue;
            }
            group.remove();
        }
    }

    function pastePanelAsEditable(targetDocument, panelPath, left, top, width, height) {
        var source = app.open(new File(panelPath));
        var sourceArtboard = source.artboards[0].artboardRect;
        var sourceLeft = sourceArtboard[0];
        var sourceTop = sourceArtboard[1];
        var sourceRight = sourceArtboard[2];
        var sourceBottom = sourceArtboard[3];
        var sourceWidth = sourceRight - sourceLeft;
        var sourceHeight = sourceTop - sourceBottom;
        var scaleX = width / sourceWidth;
        var scaleY = height / sourceHeight;
        var targetArtboard = targetDocument.artboards[0].artboardRect;
        var pasteLeft = targetArtboard[0] + sourceLeft;
        var pasteTop = targetArtboard[1];

        app.activeDocument = targetDocument;
        var group = targetDocument.groupItems.add();
        group.name = "editable_" + new File(panelPath).displayName.replace(/\\.pdf$/i, "");

        var background = group.pathItems.rectangle(pasteTop, pasteLeft, sourceWidth, sourceHeight);
        background.name = "panel_background";
        background.filled = true;
        background.fillColor = rgb(255, 255, 255);
        background.stroked = false;
        background.zOrder(ZOrderMethod.SENDTOBACK);

        app.activeDocument = source;
        source.selection = null;
        app.executeMenuCommand("selectall");
        app.copy();

        app.activeDocument = targetDocument;
        targetDocument.selection = null;
        app.executeMenuCommand("pasteInPlace");
        var pasted = targetDocument.selection;
        for (var i = pasted.length - 1; i >= 0; i--) {
            pasted[i].move(group, ElementPlacement.PLACEATBEGINNING);
        }
        background.zOrder(ZOrderMethod.SENDTOBACK);

        app.activeDocument = targetDocument;
        var matrix = app.getTranslationMatrix(left - scaleX * pasteLeft, top - scaleY * pasteTop);
        matrix.mValueA = scaleX;
        matrix.mValueD = scaleY;
        matrix.mValueB = 0;
        matrix.mValueC = 0;
        group.transform(matrix, true, true, true, true, 1.0, Transformation.DOCUMENTORIGIN);
        group.zOrder(ZOrderMethod.BRINGTOFRONT);

        source.close(SaveOptions.DONOTSAVECHANGES);
    }

    function applyPastedArtboardCorrection(documentRef) {
        var sourcePanelTop = 352.8;
        var scaleY = topPanelHeight / sourcePanelTop;
        var correction = -scaleY * (documentRef.artboards[0].artboardRect[1] - sourcePanelTop);
        for (var i = 0; i < documentRef.groupItems.length; i++) {
            var group = documentRef.groupItems[i];
            if (String(group.name).indexOf("editable_") !== 0) {
                continue;
            }
            for (var j = 0; j < group.pageItems.length; j++) {
                if (group.pageItems[j].name !== "panel_background") {
                    group.pageItems[j].translate(0, correction);
                }
            }
        }
    }

    function replaceTopRowTitles(documentRef) {
        removeGeneratedTopRowTitleText(documentRef);

        var titleSize = getPanelDTitleSize(documentRef);
        var titles = [
            "Upper lifespan tail is sensitive to\rheterogeneity in senogenic parameters",
            "Late-life survival is consistent with heterogeneity\rin robustness parameters",
            "Upper lifespan tail is sensitive to\rchanges in senogenic parameters"
        ];

        for (var i = 0; i < titles.length; i++) {
            var left = topX + i * (topPanelWidth + topGap);
            addPanelTitle(documentRef, titles[i], left, topY + 12, topPanelWidth, 62, titleSize, "fig2_top_title_" + i);
        }
    }

    function removeGeneratedTopRowTitleText(documentRef) {
        var fragments = [
            "Upper lifespan tail is sensitive to",
            "heterogeneity in senogenic parameters",
            "Late-life survival is consistent with heterogeneity",
            "in robustness parameters",
            "changes in senogenic parameters"
        ];

        for (var i = documentRef.textFrames.length - 1; i >= 0; i--) {
            var textFrame = documentRef.textFrames[i];
            var contents = String(textFrame.contents).replace(/\n/g, "\r");
            if (String(textFrame.name).indexOf("fig2_top_title_") === 0 || contentsMatches(contents, fragments)) {
                textFrame.remove();
            }
        }
    }

    function contentsMatches(contents, fragments) {
        for (var i = 0; i < fragments.length; i++) {
            if (contents === fragments[i] || contents.indexOf(fragments[i] + "\r") >= 0 || contents.indexOf("\r" + fragments[i]) >= 0) {
                return true;
            }
        }
        return false;
    }

    function addPanelTitle(documentRef, contents, left, top, width, height, size, name) {
        var box = documentRef.pathItems.rectangle(top, left + 6, width - 12, height);
        box.stroked = false;
        box.filled = false;

        var textFrame = documentRef.textFrames.areaText(box);
        textFrame.name = name;
        textFrame.contents = contents;
        textFrame.textRange.characterAttributes.size = size;
        textFrame.textRange.characterAttributes.leading = size * 1.08;
        textFrame.textRange.characterAttributes.fillColor = rgb(0, 0, 0);
        setFontIfAvailable(textFrame, "ArialMT");
        textFrame.textRange.paragraphAttributes.justification = Justification.CENTER;
        textFrame.zOrder(ZOrderMethod.BRINGTOFRONT);
    }

    function getPanelDTitleSize(documentRef) {
        for (var i = 0; i < documentRef.textFrames.length; i++) {
            var textFrame = documentRef.textFrames[i];
            if (String(textFrame.contents).indexOf("Mortality converges for siblings of centenarians") >= 0) {
                return textFrame.textRange.characterAttributes.size;
            }
        }
        return 24.167179107666;
    }

    function setFontIfAvailable(textFrame, fontName) {
        try {
            textFrame.textRange.characterAttributes.textFont = app.textFonts.getByName(fontName);
        } catch (error) {}
    }

    function savePdf(documentRef, path) {
        var options = new PDFSaveOptions();
        options.preserveEditability = true;
        options.compatibility = PDFCompatibility.ACROBAT8;
        options.generateThumbnails = true;
        options.optimization = true;
        documentRef.saveAs(new File(path), options);
    }

    function rgb(red, green, blue) {
        var color = new RGBColor();
        color.red = red;
        color.green = green;
        color.blue = blue;
        return color;
    }
})();
