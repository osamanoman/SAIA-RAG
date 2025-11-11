# SAIA Document Preparation Guide

**Version**: 1.0  
**Last Updated**: 2025-10-26  
**Purpose**: Guide for preparing legal documents before importing into SAIA knowledge base

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Document Requirements](#document-requirements)
3. [Preparation Steps](#preparation-steps)
4. [Metadata Guidelines](#metadata-guidelines)
5. [Quality Checklist](#quality-checklist)
6. [Testing Strategy](#testing-strategy)
7. [Common Issues](#common-issues)
8. [Examples](#examples)

---

## Overview

### What This Guide Covers

This guide helps you prepare legal documents (laws, regulations, statutes) for import into the SAIA knowledge base to ensure:
- ✅ Optimal AI retrieval accuracy
- ✅ Consistent article citation
- ✅ High-quality responses
- ✅ Proper metadata preservation

### Current System Capabilities

SAIA automatically handles:
- **Vector embeddings** using OpenAI text-embedding-3-large (3072 dimensions)
- **Article-aware chunking** (800 tokens, 100 overlap)
- **Metadata preservation** (article_number, article_title, legal_topic, book, chapter)
- **Semantic search** with confidence scoring
- **Arabic language processing**

---

## Document Requirements

### Supported Document Types

| Type | Example | Status |
|------|---------|--------|
| **Laws** | نظام الأحوال الشخصية | ✅ Supported |
| **Regulations** | نظام الإحصاء | ✅ Supported |
| **Executive Bylaws** | اللائحة التنفيذية | ✅ Supported |
| **Royal Decrees** | مرسوم ملكي | ✅ Supported |
| **Ministerial Decisions** | قرار وزاري | ✅ Supported |

### File Format Requirements

**Accepted Formats**:
- ✅ Plain text (.txt) - **Recommended**
- ✅ Microsoft Word (.docx)
- ✅ PDF (text-based, not scanned images)
- ✅ Markdown (.md)

**Not Supported**:
- ❌ Scanned PDFs (images)
- ❌ Handwritten documents
- ❌ Audio/Video files

### Language Requirements

- **Primary**: Arabic (العربية)
- **Encoding**: UTF-8 (mandatory)
- **Diacritics**: Optional but should be consistent

---

## Preparation Steps

### Step 1: Document Collection

#### 1.1 Obtain Official Version
- ✅ Use official government sources
- ✅ Verify document authenticity
- ✅ Check for latest amendments
- ✅ Note effective date

#### 1.2 Verify Completeness
- [ ] All articles are included
- [ ] No missing sections or chapters
- [ ] Amendments are incorporated or noted
- [ ] Cross-references are intact

---

### Step 2: Content Cleaning

#### 2.1 Remove Unnecessary Elements

**Remove These**:
```
❌ Page numbers (e.g., "صفحة 1 من 10")
❌ Headers/footers (e.g., "وزارة العدل - سري")
❌ Watermarks
❌ Administrative stamps
❌ Document metadata (file properties)
❌ Table of contents (will be auto-generated)
❌ Blank lines (excessive spacing)
```

**Keep These**:
```
✅ Article numbers (المادة الأولى، الثانية، etc.)
✅ Article titles/subjects
✅ Legal text content
✅ Definitions sections
✅ Cross-references between articles
✅ Chapter/book divisions
✅ Footnotes (if legally relevant)
```

#### 2.2 Text Encoding Check

**Verify Arabic Text**:
```bash
# Check for encoding issues:
- No question marks (???)
- No boxes (□□□)
- No mixed Arabic/Latin characters
- Proper right-to-left display
```

**Example - Good Encoding**:
```
✅ المادة الأولى: التعريفات
✅ النظام: نظام الإحصاء.
```

**Example - Bad Encoding**:
```
❌ ??????? ???????: ?????????
❌ ??????: ???? ????????.
```

---

### Step 3: Structure Standardization

#### 3.1 Article Numbering Format

**Choose ONE consistent format**:

**Option A: Arabic Ordinal Numbers** (Recommended)
```
المادة الأولى: التعريفات
المادة الثانية: نطاق التطبيق
المادة الثالثة: الاختصاصات
```

**Option B: Numeric**
```
المادة 1: التعريفات
المادة 2: نطاق التطبيق
المادة 3: الاختصاصات
```

**❌ Don't Mix Formats**:
```
❌ المادة الأولى: التعريفات
❌ المادة 2: نطاق التطبيق
❌ Article 3: الاختصاصات
```

#### 3.2 Article Structure Template

```
المادة [رقم]: [عنوان المادة]

[نص المادة الرئيسي]

[إذا كانت هناك فقرات:]
1. [الفقرة الأولى]
2. [الفقرة الثانية]
3. [الفقرة الثالثة]

[إذا كانت هناك بنود فرعية:]
أ. [البند الأول]
ب. [البند الثاني]
ج. [البند الثالث]

---
```

#### 3.3 Definitions Section Format

**For Article 1 (Definitions)**:

```
المادة الأولى: التعريفات

لأغراض هذا النظام، يقصد بالألفاظ والعبارات الآتية -أينما وردت فيه- المعاني المبينة أمام كل منها، ما لم يقتضِ السياق غير ذلك:

1. النظام: [التعريف الكامل].
2. اللائحة: [التعريف الكامل].
3. الهيئة: [التعريف الكامل].
...
```

---

### Step 4: Add Contextual Information

#### 4.1 Document Header

Add this at the **beginning** of your document:

```
===========================================
[اسم النظام الكامل]
===========================================

الصادر عن: [الجهة المصدرة]
تاريخ الإصدار: [التاريخ الهجري / الميلادي]
رقم المرسوم/القرار: [إن وجد]
تاريخ النفاذ: [تاريخ بدء التطبيق]

---

نبذة عن النظام:
[وصف موجز (2-3 جمل) عن الغرض من النظام ونطاق تطبيقه]

مثال:
"هذا النظام ينظم الأعمال الإحصائية في المملكة العربية السعودية، 
ويحدد اختصاصات الهيئة العامة للإحصاء، وآليات جمع البيانات 
والمعلومات الإحصائية من الجهات العامة والمنشآت الخاصة."

===========================================
```

#### 4.2 Chapter/Book Divisions

If the law has multiple books or chapters:

```
===========================================
الباب الأول: أحكام عامة
===========================================

المادة الأولى: التعريفات
[محتوى المادة]

---

المادة الثانية: نطاق التطبيق
[محتوى المادة]

---

===========================================
الباب الثاني: الاختصاصات والصلاحيات
===========================================

المادة الثالثة: اختصاصات الهيئة
[محتوى المادة]

---
```

---

## Metadata Guidelines

### Required Metadata

When uploading through the document management interface, provide:

| Field | Description | Example |
|-------|-------------|---------|
| **Title** | Full official name | `نظام الإحصاء` |
| **Category** | Legal domain | `statistics_law` or `administrative_law` |
| **Document Type** | Type of legal document | `law`, `regulation`, `decree` |

### Optional Metadata (Recommended)

| Field | Description | Example |
|-------|-------------|---------|
| **Issuing Authority** | Government body | `الهيئة العامة للإحصاء` |
| **Issue Year** | Year of issuance | `2020` |
| **Effective Date** | When it came into force | `1441-05-15` (Hijri) |
| **Total Articles** | Number of articles | `45` |
| **Language** | Primary language | `Arabic` |
| **Tags** | Keywords for search | `["statistics", "data", "census"]` |

### Category Guidelines

**Choose the most specific category**:

```
Family Law:
- personal_status_law (الأحوال الشخصية)
- marriage_law (الزواج)
- divorce_law (الطلاق)
- custody_law (الحضانة)
- inheritance_law (الميراث)

Administrative Law:
- statistics_law (الإحصاء)
- civil_service_law (الخدمة المدنية)
- government_procurement (المشتريات الحكومية)

Commercial Law:
- companies_law (الشركات)
- commercial_transactions (المعاملات التجارية)
- bankruptcy_law (الإفلاس)

Criminal Law:
- penal_code (العقوبات)
- criminal_procedures (الإجراءات الجزائية)
```

---

## Quality Checklist

### Pre-Import Checklist

Use this checklist before uploading any document:

#### ✅ Content Quality
- [ ] All articles are present and in order
- [ ] No missing sections or chapters
- [ ] Article numbering is sequential
- [ ] Cross-references are intact
- [ ] Definitions are complete

#### ✅ Formatting
- [ ] Consistent article numbering format
- [ ] Clear article titles/subjects
- [ ] Proper paragraph/clause numbering
- [ ] Consistent use of separators (---)
- [ ] No excessive blank lines

#### ✅ Text Quality
- [ ] Arabic text displays correctly (UTF-8)
- [ ] No encoding errors (???, □□□)
- [ ] Diacritics are consistent (all or none)
- [ ] No mixed language issues
- [ ] Proper right-to-left formatting

#### ✅ Metadata
- [ ] Document title is accurate
- [ ] Category is appropriate
- [ ] Issuing authority is noted
- [ ] Issue/effective dates are included
- [ ] Tags are relevant

#### ✅ Cleanup
- [ ] Page numbers removed
- [ ] Headers/footers removed
- [ ] Watermarks removed
- [ ] Administrative notes removed
- [ ] Irrelevant metadata removed

---

## Testing Strategy

### Post-Import Testing

After importing a document, test with these question types:

#### 1. Basic Retrieval Tests

**Test article-specific queries**:
```
Q: ما هو تعريف [مصطلح] في النظام؟
Expected: Should cite Article 1 (Definitions)

Q: ما هي المادة [رقم] من النظام؟
Expected: Should return exact article content

Q: ما هو موضوع المادة [رقم]؟
Expected: Should return article title and summary
```

#### 2. Concept-Based Tests

**Test understanding of legal concepts**:
```
Q: ما هي شروط [موضوع قانوني]؟
Expected: Should cite relevant articles with conditions

Q: كيف يتم [إجراء قانوني]؟
Expected: Should cite procedural articles

Q: من المسؤول عن [مهمة/اختصاص]؟
Expected: Should cite authority/jurisdiction articles
```

#### 3. Comparison Tests

**Test ability to distinguish concepts**:
```
Q: ما الفرق بين [مصطلح أ] و [مصطلح ب]؟
Expected: Should cite definitions and explain differences

Q: متى يطبق [حكم أ] ومتى يطبق [حكم ب]؟
Expected: Should cite conditions and contexts
```

#### 4. Cross-Reference Tests

**Test article relationships**:
```
Q: ما هي المواد المتعلقة بـ [موضوع]؟
Expected: Should cite multiple related articles

Q: هل هناك استثناءات لـ [حكم]؟
Expected: Should cite exception articles if they exist
```

### Expected Performance Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Confidence Score** | > 0.35 | For relevant queries |
| **Article Citation** | 100% | Should always cite article numbers |
| **Response Time** | < 5 seconds | For standard queries |
| **Accuracy** | > 90% | Correct article references |

---

## Common Issues

### Issue 1: Poor Retrieval Accuracy

**Symptoms**:
- AI can't find relevant articles
- Low confidence scores (< 0.30)
- Generic responses without citations

**Causes**:
- ❌ Inconsistent article formatting
- ❌ Missing article numbers
- ❌ Poor metadata
- ❌ Mixed encoding

**Solutions**:
1. Verify article numbering is consistent
2. Add contextual introduction
3. Ensure UTF-8 encoding
4. Provide rich metadata

---

### Issue 2: Incorrect Article Citations

**Symptoms**:
- AI cites wrong article numbers
- Mixes up similar articles
- Can't distinguish between concepts

**Causes**:
- ❌ Duplicate article numbers
- ❌ Unclear article titles
- ❌ Similar content in multiple articles

**Solutions**:
1. Ensure unique article numbers
2. Add descriptive article titles
3. Include chapter/book divisions
4. Test with disambiguation queries

---

### Issue 3: Encoding Problems

**Symptoms**:
- Arabic text shows as ??? or □□□
- Mixed Arabic/Latin characters
- Broken diacritics

**Causes**:
- ❌ Non-UTF-8 encoding
- ❌ Copy-paste from incompatible sources
- ❌ File format conversion issues

**Solutions**:
1. Convert to UTF-8 before upload
2. Use plain text editor (not Word)
3. Verify display before upload
4. Re-type if necessary

---

## Examples

### Example 1: Statistics Law (نظام الإحصاء)

**Original Document** (needs preparation):
```
نظام الإحصاء
المادة الأولى
لأغراض هذا النظام، يقصد بالألفاظ والعبارات الآتية...
النظام: نظام الإحصاء.
اللائحة: اللائحة التنفيذية للنظام.
```

**Prepared Document** (ready for import):
```
===========================================
نظام الإحصاء
===========================================

الصادر عن: الهيئة العامة للإحصاء
تاريخ الإصدار: 1441هـ / 2020م
رقم المرسوم: [رقم المرسوم الملكي]
تاريخ النفاذ: [تاريخ بدء التطبيق]

---

نبذة عن النظام:
هذا النظام ينظم الأعمال الإحصائية في المملكة العربية السعودية، 
ويحدد اختصاصات الهيئة العامة للإحصاء، وآليات جمع البيانات 
والمعلومات الإحصائية من الجهات العامة والمنشآت الخاصة.

===========================================

الباب الأول: أحكام عامة
===========================================

المادة الأولى: التعريفات

لأغراض هذا النظام، يقصد بالألفاظ والعبارات الآتية -أينما وردت فيه- 
المعاني المبينة أمام كل منها، ما لم يقتضِ السياق غير ذلك:

1. النظام: نظام الإحصاء.
2. اللائحة: اللائحة التنفيذية للنظام.
3. الهيئة: الهيئة العامة للإحصاء.
4. المجلس: مجلس إدارة الهيئة.
...

---

المادة الثانية: نطاق التطبيق
[محتوى المادة]

---
```

**Metadata to Provide**:
```json
{
  "title": "نظام الإحصاء",
  "category": "statistics_law",
  "document_type": "law",
  "issuing_authority": "الهيئة العامة للإحصاء",
  "issue_year": "2020",
  "language": "Arabic",
  "tags": ["statistics", "data", "census", "administrative law", "saudi arabia"]
}
```

---

## Final Recommendations

### Best Practices

1. **Start Small**: Import one complete law first, test thoroughly
2. **Be Consistent**: Use same formatting across all documents
3. **Test Incrementally**: Import → Test → Refine → Repeat
4. **Document Changes**: Keep notes on amendments and updates
5. **Verify Sources**: Always use official government sources

### Quality Over Quantity

- ✅ One well-prepared document > Multiple poorly formatted documents
- ✅ Complete articles > Partial content
- ✅ Accurate metadata > No metadata
- ✅ Tested retrieval > Untested import

---

**Document prepared according to this guide will ensure optimal AI performance and accurate legal consultation.** 🎯

