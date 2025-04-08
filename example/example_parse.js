// --- n8n Code Node Script (Improved Regex Version - No External Libs) ---
// WARNING: This regex-based approach is fragile and highly dependent
// on the exact HTML structure provided in the example.
// It may break if the source arXiv HTML changes significantly.

function parseArxivNewSubmissionsWithRegexDetailed() {
    // --- Configuration ---
    const BASE_URL = "https://arxiv.org"; // Base URL for relative links
  
    // --- Input ---
    let htmlString = "";
    try {
      htmlString = $input.first().json.data;
      if (typeof htmlString !== 'string') {
        throw new Error("Input data is not a string.");
      }
    } catch (e) {
      console.error("Error accessing input data:", e);
      if (typeof $throw !== 'undefined' && typeof $throw.error === 'function') {
        $throw.error(`Could not read input data: ${e.message}`);
      }
      return { submissions: [], error: `Could not read input data: ${e.message}` };
    }
  
    const results = []; // Array to hold the parsed submission objects
  
    if (!htmlString || htmlString.trim() === '') {
      console.log("Error: Invalid or empty input HTML string.");
      return { submissions: [], error: "Invalid or empty input HTML." };
    }
    console.log(`Input HTML received. Length: ${htmlString.length}`);
  
    try {
      const articlesDlRegex = /<dl\s+id=['"]articles['"][^>]*>([\s\S]*?)<\/dl>/i;
      const articlesDlMatch = htmlString.match(articlesDlRegex);
      if (!articlesDlMatch || !articlesDlMatch[1]) {
        console.log("Error: Could not find or extract content from <dl id='articles'> section.");
        return { submissions: [], error: "Could not find <dl id='articles'> section." };
      }
      const articlesHtml = articlesDlMatch[1];
      console.log("Successfully extracted content within <dl id='articles'>.");
  
      const newSubmissionsHeadingRegex = /<h3[^>]*>\s*New submissions[\s\S]*?<\/h3>/i;
      let headingMatch = articlesHtml.match(newSubmissionsHeadingRegex);
      let startIndex = 0; // Default to start if heading not found
  
      if (headingMatch && typeof headingMatch.index === 'number') {
         startIndex = headingMatch.index + headingMatch[0].length;
         console.log(`Found 'New submissions' heading. Start index for content: ${startIndex}`);
      } else {
         console.log("Warning: 'New submissions' heading not found. Assuming content starts from the beginning of <dl>.");
         // Optional: Fallback to first <dt> if needed, but starting from 0 might be safer if heading is missing
      }
  
  
      const contentAfterHeading = articlesHtml.substring(startIndex);
      const nextHeadingRegex = /<h3/i;
      const nextHeadingMatch = contentAfterHeading.match(nextHeadingRegex);
  
      let endIndex = articlesHtml.length; // Default to end if no next heading
      if (nextHeadingMatch && typeof nextHeadingMatch.index === 'number') {
        endIndex = startIndex + nextHeadingMatch.index;
        console.log(`Found next <h3> tag. Absolute end index: ${endIndex}`);
      } else {
        console.log("No subsequent <h3> found, using end of <dl> content as end index.");
      }
  
      const newSubmissionsHtml = articlesHtml.substring(startIndex, endIndex);
      if (!newSubmissionsHtml || newSubmissionsHtml.trim().length === 0) {
        console.log("Warning: Extracted newSubmissionsHtml section is empty.");
        return { submissions: [], error: "New submissions section appears empty." };
      }
      console.log(`Extracted newSubmissionsHtml length: ${newSubmissionsHtml.length}. Starting extraction.`);
  
      const dtDdPairRegex = /<dt[^>]*>([\s\S]*?)<\/dt>\s*<dd[^>]*>([\s\S]*?)<\/dd>/gis;
  
      // --- Regexes for extracting specific data ---
      // Regex to capture the main arXiv link, its href path, and the ID within the path.
      // It specifically looks for href starting with "/abs/" and having title="Abstract"
      const arxivLinkRegex = /<a\s+href=["'](\/abs\/([^"']+))["'][^>]*title=["']Abstract["']/i;
  
      const pdfLinkRegex = /<a\s+href=["'](\/pdf\/[^"']+)["'][^>]*title=["']Download PDF["']/i;
      const htmlLinkRegex = /<a\s+href=["']([^"']+)["'][^>]*title=["']View HTML["']/i;
  
      const titleRegex = /<div class=["']list-title mathjax["']>\s*<span class=["']descriptor["']>Title:<\/span>\s*([\s\S]*?)<\/div>/is;
      const authorsRegex = /<div class=["']list-authors["']>([\s\S]*?)<\/div>/is;
      const authorLinkRegex = /<a\s+href=["'][^"']+["'][^>]*>([\s\S]*?)<\/a>/gi;
      const commentsRegex = /<div class=["']list-comments mathjax["']>\s*<span class=["']descriptor["']>Comments:<\/span>\s*([\s\S]*?)<\/div>/is;
      const subjectsRegex = /<div class=["']list-subjects["']>\s*<span class=["']descriptor["']>Subjects:<\/span>\s*([\s\S]*?)<\/div>/is;
      const primarySubjectRegex = /<span class=["']primary-subject["'][^>]*>([\s\S]*?)<\/span>/i;
  
      const cleanText = (htmlSnippet) => {
        if (!htmlSnippet) return null;
        return htmlSnippet.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
      };
  
      let match;
      let pairsFound = 0;
  
      console.log("Starting loop to find DT/DD pairs...");
      while ((match = dtDdPairRegex.exec(newSubmissionsHtml)) !== null) {
        pairsFound++;
        const dtContent = match[1];
        const ddContent = match[2];
  
        let titleText = null;
        let authorsText = null;
        let commentsText = null;
        let subjectsText = null;
        let primarySubjectText = null;
        let abstractUrl = null;
        let htmlUrl = null;
        let pdfUrl = null;
        let arxivId = null;
  
        // --- Extract from DT: ID and Abstract URL first ---
        const arxivMatch = dtContent.match(arxivLinkRegex);
        if (arxivMatch && arxivMatch[1] && arxivMatch[2]) {
          const relativeAbsPath = arxivMatch[1];
          arxivId = arxivMatch[2]; // ID extracted from the path
          abstractUrl = `${BASE_URL}${relativeAbsPath}`; // Construct full URL
        } else {
           // Fallback if the specific abstract link format isn't found
           const fallbackIdTextMatch = dtContent.match(/arXiv:([\d.]+)/i);
           if (fallbackIdTextMatch && fallbackIdTextMatch[1]) {
               arxivId = fallbackIdTextMatch[1];
               // Construct abstract URL from text ID if available
               abstractUrl = `${BASE_URL}/abs/${arxivId}`;
               console.log(`   Warning: Used fallback to find text ID ${arxivId} and constructed abstract URL for pair ${pairsFound}.`);
           } else {
               console.log(`   ERROR: Could not find arXiv ID or abstract link for pair ${pairsFound}. Skipping.`);
               continue; // Skip this entry if no ID can be found
           }
        }
  
        // Extract other URLs using the BASE_URL logic
        const pdfMatch = dtContent.match(pdfLinkRegex);
        if (pdfMatch && pdfMatch[1]) {
          pdfUrl = pdfMatch[1].startsWith('/') ? `${BASE_URL}${pdfMatch[1]}` : pdfMatch[1];
        }
  
        const htmlMatch = dtContent.match(htmlLinkRegex);
        if (htmlMatch && htmlMatch[1]) {
          htmlUrl = htmlMatch[1].startsWith('/') ? `${BASE_URL}${htmlMatch[1]}` : htmlMatch[1];
        }
  
        // --- Extract from DD ---
        const titleMatch = ddContent.match(titleRegex);
        if (titleMatch && titleMatch[1]) {
          titleText = cleanText(titleMatch[1]);
        } else {
          console.log(`   Warning: No title found for pair ${pairsFound} (ID: ${arxivId})`);
          // Decide if missing title is critical; currently allows proceeding
        }
  
        const authorsMatch = ddContent.match(authorsRegex);
        if (authorsMatch && authorsMatch[1]) {
          const authorNames = [];
          let authorLinkMatch;
          while ((authorLinkMatch = authorLinkRegex.exec(authorsMatch[1])) !== null) {
            authorNames.push(cleanText(authorLinkMatch[1]));
          }
          authorsText = authorNames.length > 0 ? authorNames.join(', ') : cleanText(authorsMatch[1]);
        }
  
        const commentsMatch = ddContent.match(commentsRegex);
        if (commentsMatch && commentsMatch[1]) {
          commentsText = cleanText(commentsMatch[1]);
        }
  
        const subjectsMatch = ddContent.match(subjectsRegex);
        if (subjectsMatch && subjectsMatch[1]) {
          const subjectsContent = subjectsMatch[1]; // Keep the raw inner HTML
          subjectsText = cleanText(subjectsContent); // Cleaned version for the 'subjects' field
          const primarySubjectMatch = subjectsContent.match(primarySubjectRegex);
          if (primarySubjectMatch && primarySubjectMatch[1]) {
            primarySubjectText = cleanText(primarySubjectMatch[1]);
          }
        }
  
        // Create the submission data object (ensure critical fields like ID are present)
        const submissionData = {
          arxiv_id: arxivId, // Should always be populated if we didn't 'continue'
          title: titleText, // Might be null if regex failed
          abstract_url: abstractUrl, // Should be populated if ID was found
          pdf_url: pdfUrl,
          html_url: htmlUrl,
          authors: authorsText,
          subjects: subjectsText,
          primary_subject: primarySubjectText,
          comments: commentsText,
        };
        results.push(submissionData);
  
        if (pairsFound <= 5) { // Log first few successful extractions
          console.log(`   Processed Pair ${pairsFound}: ID=${arxivId}, Title=${titleText ? titleText.substring(0, 30) + '...' : 'N/A'}, AbstractURL=${abstractUrl}`);
        }
  
      } // End while loop
  
      if (pairsFound === 0) {
          console.log("Warning: Loop finished, but no DT/DD pairs were matched in the 'New submissions' section.");
      } else {
          console.log(`Finished processing. Found ${pairsFound} DT/DD pairs. Successfully extracted ${results.length} submissions.`);
      }
  
    } catch (error) {
      console.error("Error during HTML processing with regex:", error);
       if (typeof $throw !== 'undefined' && typeof $throw.error === 'function') {
        $throw.error(`Processing error: ${error.message}`);
      }
      return { submissions: [], error: `Processing error: ${error.message}` };
    }
  
    return { submissions: results };
  }
  
  // --- n8n Code Node Execution ---
  const parsedData = parseArxivNewSubmissionsWithRegexDetailed();
  return [{ json: parsedData }];