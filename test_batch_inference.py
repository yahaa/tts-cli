#!/usr/bin/env python3
"""Test script to verify ChatTTS batch inference stability."""

import time
import ChatTTS

def test_batch_inference():
    print("=" * 60)
    print("ChatTTS Batch Inference Test")
    print("=" * 60)

    # Initialize ChatTTS
    print("\n[1] Loading ChatTTS model...")
    chat = ChatTTS.Chat()
    chat.load()
    print("    Model loaded successfully!")

    # Test texts - pure English
    english_texts = [
        "Hello world, this is test one.",
        "Another test with different length here for comparison.",
        "Short test.",
        "This is a longer test sentence to see if length matters in batch processing.",
    ]

    # Test texts - pure Chinese
    chinese_texts = [
        "你好世界，这是第一个测试。",
        "这是另一个测试，长度稍微不同一些。",
        "短测试。",
        "这是一个更长的测试句子，看看长度是否会影响批量处理的结果。",
    ]

    # Test texts - mixed
    mixed_texts = [
        "Hello world, this is English.",
        "你好世界，这是中文。",
        "This is another English sentence.",
        "这是另一个中文句子。",
    ]

    # Sample speaker for consistency
    print("\n[2] Sampling speaker embedding...")
    spk = chat.sample_random_speaker()

    params = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk,
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )

    def run_batch_test(name, texts):
        print(f"\n[Test: {name}]")
        print(f"    Input: {len(texts)} texts")
        for i, t in enumerate(texts):
            print(f"      [{i}] {t[:50]}...")

        start = time.time()
        try:
            wavs = chat.infer(
                texts,
                params_infer_code=params,
                skip_refine_text=True,
            )
            elapsed = time.time() - start

            if wavs is None:
                print(f"    ❌ FAILED: wavs is None")
                return False

            if len(wavs) != len(texts):
                print(f"    ❌ FAILED: Expected {len(texts)} outputs, got {len(wavs)}")
                return False

            success_count = sum(1 for w in wavs if w is not None and len(w) > 0)
            if success_count == len(texts):
                print(f"    ✅ SUCCESS: {success_count}/{len(texts)} audios generated in {elapsed:.2f}s")
                return True
            else:
                print(f"    ⚠️ PARTIAL: {success_count}/{len(texts)} audios generated in {elapsed:.2f}s")
                for i, w in enumerate(wavs):
                    status = "OK" if w is not None and len(w) > 0 else "EMPTY"
                    print(f"      [{i}] {status}")
                return False

        except Exception as e:
            elapsed = time.time() - start
            print(f"    ❌ ERROR after {elapsed:.2f}s: {e}")
            return False

    # Run tests
    results = {}
    results['English'] = run_batch_test("Pure English", english_texts)
    results['Chinese'] = run_batch_test("Pure Chinese", chinese_texts)
    results['Mixed'] = run_batch_test("Mixed EN/ZH", mixed_texts)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")

    all_pass = all(results.values())
    print(f"\nOverall: {'✅ All tests passed!' if all_pass else '❌ Some tests failed'}")
    return all_pass

if __name__ == "__main__":
    test_batch_inference()
