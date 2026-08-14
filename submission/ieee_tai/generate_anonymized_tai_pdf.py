import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, FrameBreak,
    Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib import colors

def build_pdf():
    base_dir = "./submission/ieee_tai"
    pdf_path = os.path.join(base_dir, "Anonymized_Main_Document.pdf")

    margin = 36 # 0.5 inch margins for IEEE standard
    doc = BaseDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin
    )

    page_width, page_height = letter
    content_width = page_width - 2 * margin
    content_height = page_height - 2 * margin
    
    col_gap = 14
    col_width = (content_width - col_gap) / 2.0

    # Top frame for title & abstract (full width), two column frames below
    title_height = 205
    frame_top = Frame(margin, page_height - margin - title_height, content_width, title_height, id='top_frame', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    frame_c1_first = Frame(margin, margin, col_width, content_height - title_height - 10, id='c1_first', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    frame_c2_first = Frame(margin + col_width + col_gap, margin, col_width, content_height - title_height - 10, id='c2_first', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)

    # Subsequent pages: two full-height columns
    frame_c1_later = Frame(margin, margin, col_width, content_height, id='c1_later', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    frame_c2_later = Frame(margin + col_width + col_gap, margin, col_width, content_height, id='c2_later', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)

    first_page_template = PageTemplate(id='FirstPage', frames=[frame_top, frame_c1_first, frame_c2_first])
    later_page_template = PageTemplate(id='LaterPages', frames=[frame_c1_later, frame_c2_later])

    doc.addPageTemplates([first_page_template, later_page_template])

    styles = getSampleStyleSheet()

    # Typography styles matching IEEE Transactions
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=17,
        alignment=1, # Center
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=5
    )

    anon_author_style = ParagraphStyle(
        'AnonAuthor',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=11,
        alignment=1, # Center
        textColor=colors.HexColor('#475569'),
        spaceAfter=6
    )

    running_head_style = ParagraphStyle(
        'RunningHead',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=6
    )

    abs_head_style = ParagraphStyle(
        'AbsHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=2
    )

    abs_body_style = ParagraphStyle(
        'AbsBody',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.8,
        leading=10.2,
        alignment=4, # Justified
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=5
    )

    impact_style = ParagraphStyle(
        'ImpactBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.8,
        alignment=4, # Justified
        textColor=colors.HexColor('#334155'),
        spaceAfter=5
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.8,
        leading=11,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=7,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.2,
        leading=10.5,
        alignment=4, # Justified
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )

    eq_style = ParagraphStyle(
        'Equation',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.2,
        leading=10.5,
        alignment=1, # Center
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=2,
        spaceAfter=2
    )

    tbl_text = ParagraphStyle(
        'TblText',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7,
        leading=8.8,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    tbl_header = ParagraphStyle(
        'TblHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=8.8,
        alignment=1,
        textColor=colors.white
    )

    story = []

    # Running Header
    story.append(Paragraph("IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE — DOUBLE-ANONYMOUS PEER REVIEW MANUSCRIPT", running_head_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=6))

    # Title & Anonymized Author
    story.append(Paragraph("When Confidence Proxies Confound Reasoning Complexity:<br/>Pitfalls of Uncertainty-Weighted Credit Assignment in Language Model Reinforcement Learning", title_style))
    story.append(Paragraph("Anonymous Author(s) &bull; Double-Blind Peer Review Submission", anon_author_style))

    # Abstract & Impact Statement in Header Frame
    story.append(Paragraph("<b>Abstract</b>—Reinforcement learning from rule-based verifiers (RLVR), notably Group Relative Policy Optimization (GRPO), has emerged as a cornerstone for post-training Large Language Models (LLMs) on complex reasoning tasks. A natural hypothesis is that regularizing policy gradient advantages by trajectory-level uncertainty could prevent policy collapse and filter noisy exploration traces. In this work, we conduct a systematic empirical and architectural audit of uncertainty-regularized GRPO. First, we show that Monte Carlo (MC) dropout probing becomes degenerate when the target architecture lacks active dropout in its evaluated forward graph, as confirmed on Qwen2.5. Second, through diagnostic evaluations on untouched GSM8K and SVAMP datasets (N=100), we find that internal confidence proxies—token predictive entropy (r = +0.486), mean token negative log-likelihood (r = +0.432), and logit margins (r = +0.495)—are strongly correlated with derivation length and arithmetic step complexity. In stress tests, internal token entropy misidentifies correct multi-step reasoning traces as more uncertain than short incorrect errors in 42.1% of paired comparisons. While external consensus disagreement (Self-Consistency) provides unconfounded offline error discrimination (AUROC = 0.812), a preregistered 5-way controlled RL experiment demonstrates that using consensus weights for online advantage scaling yields no performance improvement over standard outcome-supervised GRPO (80.00% vs 80.00%; Delta = 0.00%). Our findings demonstrate that high offline error-predictive validity does not guarantee online reinforcement learning credit utility, establishing the necessity of negative controls when evaluating confidence-guided post-training algorithms.", abs_body_style))
    
    story.append(Paragraph("<b>Impact Statement</b>—Reinforcement learning algorithms for Large Language Models increasingly govern automated reasoning systems in mathematics, coding, and decision support. Integrating confidence or uncertainty weights into policy gradient updates is frequently proposed to prevent reward hacking and model drift. This work provides a rigorous methodological warning: internal uncertainty signals frequently track legitimate reasoning complexity rather than error, causing naive uncertainty-weighted algorithms to suppress valid multi-step exploration. By establishing that offline error predictors do not automatically improve online policy learning, our study provides a reproducible diagnostic framework and negative-control protocol to prevent the adoption of unverified or counterproductive credit assignment mechanisms.", impact_style))
    
    story.append(Paragraph("<b>Index Terms</b>—Reinforcement learning, Group Relative Policy Optimization (GRPO), Large Language Models, uncertainty estimation, self-consistency, reasoning complexity, post-training.", abs_head_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=5))
    story.append(FrameBreak()) # Break to Column 1

    # SECTION I: INTRODUCTION
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Post-training autoregressive Large Language Models (LLMs) using reinforcement learning from verifiable rewards (RLVR) has achieved substantial milestones in mathematical derivation and formal deduction [1, 2]. Group Relative Policy Optimization (GRPO) [1] estimates baseline-free advantages by normalizing outcome rewards across a group of sampled rollouts:", body_style))
    story.append(Paragraph("<i>A_hat_i = (R(y_i) - mu_R) / (sigma_R + eps)</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1)", eq_style))
    story.append(Paragraph("Because policy gradient methods can suffer from policy collapse and reward sparsity, numerous approaches hypothesize that weighting advantages by trajectory-level uncertainty <i>U(y_i)</i> can protect the policy from destabilizing updates [3, 4].", body_style))
    story.append(Paragraph("In this paper, we report a rigorous empirical and architectural investigation into uncertainty-weighted credit assignment for GRPO. We evaluate three central questions:<br/>"
                           "1. <b>Architectural Validity (C1)</b>: Does MC-dropout probing produce meaningful stochastic representations on zero-dropout architectures?<br/>"
                           "2. <b>Diagnostic Validity (C2)</b>: Do internal token-level confidence proxies distinguish mathematical errors from legitimate reasoning complexity?<br/>"
                           "3. <b>Algorithmic Validity (C3)</b>: Does a validated offline error predictor (Self-Consistency [5]) improve online GRPO policy optimization over outcome-supervised baselines and negative controls?", body_style))

    # SECTION II: ESTIMATOR VALIDITY
    story.append(Paragraph("II. ESTIMATOR VALIDITY ON ZERO-DROPOUT ARCHITECTURES", h1_style))
    story.append(Paragraph("We audited the internal layer structure of open-weight causal language models, specifically inspecting Qwen/Qwen2.5-0.5B-Instruct (Qwen2ForCausalLM). We find that the model contains exactly <b>0 active nn.Dropout modules</b> in its attention and MLP blocks (attention_dropout = 0.0). Consequently, executing Monte Carlo dropout forward passes [6] with mc_dropout=True produces mathematically deterministic passes (Var(log P) = 0.0000000000, Delta Logit = 0.0).", body_style))
    story.append(Paragraph("When normalized in advantage scaling equations with stability constant eps = 10^-8, floating-point order-of-operations noise (approx 10^-12) results in a multiplier of exp(-gamma * 10^-4) approx 0.999965, producing policy updates collinear to standard GRPO (cos(Delta theta) = 1.000000). We conclude that MC-dropout uncertainty estimation becomes degenerate when the target architecture contains no active dropout in the evaluated forward computation graph.", body_style))

    story.append(FrameBreak()) # Break to Column 2

    # SECTION III: CONFIDENCE PROXIES
    story.append(Paragraph("III. CONFIDENCE PROXIES AND REASONING COMPLEXITY", h1_style))
    story.append(Paragraph("Evaluating candidate uncertainty proxies across untouched GSM8K confirmatory data (<i>N</i> = 100 independent prompt clusters, 98 degrees of freedom) reveals a pattern consistent with a reasoning-complexity confound:<br/>"
                           "&bull; <b>Token Predictive Entropy</b> correlates strongly with sequence length (<i>r</i> = +0.486, 95% CI [+0.318, +0.627]) and equation count (<i>r</i> = +0.421).<br/>"
                           "&bull; After controlling for completion length via partial correlation, the association between token entropy and correctness collapses from <i>r</i> = -0.214 to partial <i>r</i> = -0.092 (<i>p</i> = 0.365).<br/>"
                           "&bull; In the <b>Correct-but-Complex Stress Test</b>, token entropy misidentifies correct multi-step reasoning traces as more uncertain than short incorrect errors in <b>42.1% of paired comparisons</b>.<br/>"
                           "&bull; <b>Self-Consistency</b> (<i>U_SC</i> = 1 - modal count / <i>K</i>) [5] remains robust to length bias (<i>r</i> = +0.114, partial <i>r</i> = -0.569, <i>p</i> = 8.1 x 10^-10, AUROC = 0.812).", body_style))

    # TABLE I: DIAGNOSTIC PROXY BENCHMARK
    t1_data = [
        [Paragraph("Candidate Proxy", tbl_header), Paragraph("AUROC", tbl_header), Paragraph("r(Error)", tbl_header), Paragraph("r(Length)", tbl_header), Paragraph("Partial r", tbl_header)],
        [Paragraph("Self-Consistency (K=4)", tbl_text), Paragraph("0.812", tbl_text), Paragraph("+0.582", tbl_text), Paragraph("+0.114", tbl_text), Paragraph("-0.569 (p<1e-9)", tbl_text)],
        [Paragraph("Token Entropy", tbl_text), Paragraph("0.618", tbl_text), Paragraph("+0.214", tbl_text), Paragraph("+0.486", tbl_text), Paragraph("-0.092 (p=0.365)", tbl_text)],
        [Paragraph("Mean Token NLL", tbl_text), Paragraph("0.605", tbl_text), Paragraph("+0.198", tbl_text), Paragraph("+0.432", tbl_text), Paragraph("-0.081 (p=0.422)", tbl_text)],
        [Paragraph("Logit Margin", tbl_text), Paragraph("0.624", tbl_text), Paragraph("+0.226", tbl_text), Paragraph("+0.495", tbl_text), Paragraph("-0.104 (p=0.303)", tbl_text)]
    ]
    t1 = Table(t1_data, colWidths=[col_width*0.35, col_width*0.15, col_width*0.16, col_width*0.16, col_width*0.18])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(Spacer(1, 2))
    story.append(Paragraph("TABLE I: Uncertainty Proxy Benchmark (GSM8K N=100)", abs_head_style))
    story.append(t1)
    story.append(Spacer(1, 4))

    # SECTION IV: PREREGISTERED RL EXPERIMENT
    story.append(Paragraph("IV. PREREGISTERED CONSISTENCY-AWARE RL EXPERIMENT", h1_style))
    story.append(Paragraph("To test whether validated offline self-consistency translates to an effective online policy gradient weight, we preregistered Consistency-Aware GRPO (CA-GRPO):", body_style))
    story.append(Paragraph("<i>A_tilde_i = A_hat_i * (1.0 + lambda * (c_i - c_mean))</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2)", eq_style))
    story.append(Paragraph("where <i>c_i = 1 if extract(y_i) = modal answer else 0</i>.<br/>"
                           "We benchmarked CA-GRPO against four matched controls across three independent training seeds on held-out test data evaluated under the canonical 256-token generation budget:", body_style))

    # TABLE II: RL EXPERIMENT MATRIX
    t2_data = [
        [Paragraph("Method", tbl_header), Paragraph("G", tbl_header), Paragraph("Held-Out Pass@1", tbl_header), Paragraph("Reward", tbl_header)],
        [Paragraph("Standard-GRPO", tbl_text), Paragraph("4", tbl_text), Paragraph("80.00 +/- 0.00%", tbl_text), Paragraph("0.12", tbl_text)],
        [Paragraph("Compute-Matched-GRPO", tbl_text), Paragraph("8", tbl_text), Paragraph("78.33 +/- 2.89%", tbl_text), Paragraph("0.26", tbl_text)],
        [Paragraph("Random-Weight-Control", tbl_text), Paragraph("4", tbl_text), Paragraph("75.00 +/- 5.00%", tbl_text), Paragraph("0.21", tbl_text)],
        [Paragraph("Permuted-Control", tbl_text), Paragraph("4", tbl_text), Paragraph("80.00 +/- 0.00%", tbl_text), Paragraph("0.29", tbl_text)],
        [Paragraph("<b>CA-GRPO (Proposed)</b>", tbl_text), Paragraph("4", tbl_text), Paragraph("<b>80.00 +/- 0.00%</b>", tbl_text), Paragraph("0.12", tbl_text)]
    ]
    t2 = Table(t2_data, colWidths=[col_width*0.42, col_width*0.12, col_width*0.32, col_width*0.14])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(Spacer(1, 2))
    story.append(Paragraph("TABLE II: Preregistered 5-Way RL Control Matrix (N=3 Seeds)", abs_head_style))
    story.append(t2)
    story.append(Spacer(1, 4))

    story.append(Paragraph("CA-GRPO did not improve performance over Standard GRPO in our tested configuration (Delta = 0.00%, Cohen's <i>d</i> = 0.000). Negative controls (permuted and random weighting) demonstrated that trajectory-specific advantage scaling provided no causal learning advantage over standard outcome-supervised GRPO.", body_style))

    # SECTION V: LIMITATIONS & CONCLUSION
    story.append(Paragraph("V. LIMITATIONS AND CONCLUSION", h1_style))
    story.append(Paragraph("<b>Limitations</b>: (1) RL evaluations used <i>N</i>=3 training seeds; (2) evaluations focused on mathematical reasoning (GSM8K, SVAMP) using Qwen2.5-0.5B-Instruct; (3) self-consistency requires extra inference compute (<i>K</i>=4 rollouts); and (4) findings evaluate specific linear advantage weighting.<br/>"
                           "<b>Conclusion</b>: Internal confidence proxies are consistent with a reasoning-complexity confound, and high offline error discrimination does not imply online RL credit utility. Rigorous negative controls and architectural audits are indispensable for sound RLVR post-training.", body_style))

    story.append(Paragraph("ACKNOWLEDGMENT", h1_style))
    story.append(Paragraph("Anonymized for double-blind peer review.", body_style))

    # REFERENCES
    story.append(Paragraph("REFERENCES", h1_style))
    refs = [
        "[1] Z. Shao, P. Wang, Q. Zhu, R. Xu, et al., \"DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,\" arXiv:2402.03300, 2024.",
        "[2] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, et al., \"Training verifiers to solve math word problems,\" arXiv:2110.14168, 2021.",
        "[3] S. Kadavath, T. Conerly, A. Askell, T. Henighan, et al., \"Language models (mostly) know what they know,\" arXiv:2207.05221, 2022.",
        "[4] L. Kuhn, Y. Gal, and S. Farquhar, \"Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation,\" in ICLR, 2023.",
        "[5] X. Wang, J. Wei, D. Schuurmans, Q. V. Le, et al., \"Self-consistency improves chain of thought reasoning in language models,\" in ICLR, 2023.",
        "[6] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,\" in ICML, 2016, pp. 1050-1059."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefText', parent=body_style, fontSize=7, leading=8.8, spaceAfter=2)))

    doc.build(story)
    print(f"Generated clean {pdf_path}")

if __name__ == "__main__":
    build_pdf()
