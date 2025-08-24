#!/usr/bin/env python3

from app.chunking_optimizer import get_chunking_optimizer
from app.vector_store import get_vector_store
from app.utils import chunk_text
import statistics

def test_chunking_comparison():
    """Compare old vs new chunking strategies."""
    
    # Initialize components
    optimizer = get_chunking_optimizer()
    vs = get_vector_store()
    
    # Get sample chunks for comparison
    points, _ = vs.client.scroll(
        collection_name=vs.collection_name,
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    
    print('=== CHUNKING STRATEGY COMPARISON ===')
    print()
    
    total_old_quality_issues = 0
    total_new_improvements = 0
    
    for i, point in enumerate(points):
        payload = point.payload
        text = payload.get('text', '')
        category = payload.get('category', 'general')
        title = payload.get('title', 'Unknown')
        
        print(f'--- Sample {i+1}: {title} ({category}) ---')
        print(f'Original text length: {len(text)} characters')
        
        # Old chunking (current method)
        old_chunks = chunk_text(text, chunk_size=800, overlap=200)
        
        # New optimized chunking
        new_chunks = optimizer.optimize_chunks(text, category, title)
        
        print(f'Old method: {len(old_chunks)} chunks')
        if old_chunks:
            old_sizes = [len(chunk) for chunk in old_chunks]
            print(f'  Sizes: {old_sizes}')
            print(f'  Avg size: {statistics.mean(old_sizes):.0f} chars')
            if len(old_sizes) > 1:
                print(f'  Size variance: {statistics.variance(old_sizes):.0f}')
            else:
                print(f'  Size variance: 0')
        
        print(f'New method: {len(new_chunks)} chunks')
        if new_chunks:
            new_sizes = [chunk.char_count for chunk in new_chunks]
            print(f'  Sizes: {new_sizes}')
            print(f'  Avg size: {statistics.mean(new_sizes):.0f} chars')
            if len(new_sizes) > 1:
                print(f'  Size variance: {statistics.variance(new_sizes):.0f}')
            else:
                print(f'  Size variance: 0')
            print(f'  Avg boundary quality: {statistics.mean([c.boundary_quality for c in new_chunks]):.2f}')
            print(f'  Context preserved: {sum(1 for c in new_chunks if c.context_preserved)}/{len(new_chunks)}')
        
        # Quality assessment
        old_quality_issues = 0
        if old_chunks:
            for chunk in old_chunks:
                if len(chunk) < 200 or len(chunk) > 1500:
                    old_quality_issues += 1
                if chunk.count('.') < 1 and chunk.count('؟') < 1:
                    old_quality_issues += 1
        
        new_quality_score = 0
        if new_chunks:
            new_quality_score = statistics.mean([c.boundary_quality for c in new_chunks])
        
        print(f'Quality comparison:')
        print(f'  Old method issues: {old_quality_issues}')
        print(f'  New method quality score: {new_quality_score:.2f}/1.0')
        
        total_old_quality_issues += old_quality_issues
        total_new_improvements += new_quality_score
        
        print()
    
    print(f'=== OVERALL COMPARISON ===')
    print(f'Total old quality issues: {total_old_quality_issues}')
    print(f'Average new quality score: {total_new_improvements/len(points):.2f}/1.0')
    
    # Calculate improvement
    old_score = max(0.1, 1.0 - (total_old_quality_issues / (len(points) * 3)))  # Normalize old score
    new_score = total_new_improvements / len(points)
    improvement = ((new_score - old_score) / old_score) * 100
    
    print(f'Improvement factor: {improvement:.1f}%')
    
    return {
        'old_quality_issues': total_old_quality_issues,
        'new_quality_score': new_score,
        'improvement_percentage': improvement,
        'samples_tested': len(points)
    }

if __name__ == "__main__":
    test_chunking_comparison()
