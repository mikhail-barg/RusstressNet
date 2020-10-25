using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace RusstressNet
{
    public class AccentModel : IDisposable
    {
        private const string VOVELS = "ёуеыаоэяиюЁУЕЫАОЭЯИЮ";
        private readonly Regex m_lineEndingRegex = new Regex(@"[…:,.?!\n]", RegexOptions.Compiled);
        private readonly Regex m_charsOnlyRegex = new Regex(@"[^а-яё'_ -]", RegexOptions.Compiled);
        //private readonly Regex m_hasTwoVovels = new Regex($@"[{VOVELS}].*[{VOVELS}]", RegexOptions.Compiled);
        private readonly char[] m_whitespaceSplit = { ' ' };

        private const int MAXLEN = 40;
        private readonly char[] m_chars = {'\'', '-', '_',
             'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к',
             'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш',
             'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё'
        };
        private readonly Dictionary<char, int> m_charIndices;

        private readonly List<HashSet<char>> m_defaultCategories = new List<HashSet<char>>() {
            StringToHashSet("0123456789"),
            StringToHashSet(" "),
            StringToHashSet(",.;:!?()\"[]@#$%^&*_+=«»"),
            StringToHashSet("ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ-\'")
        };

        private readonly InferenceSession m_session;
        private readonly int[] m_inputShape;

        public AccentModel()
        {
            this.m_charIndices = new Dictionary<char, int>();
            for (int i = 0; i < this.m_chars.Length; ++i)
            {
                this.m_charIndices[this.m_chars[i]] = i;
            }

            this.m_session = new InferenceSession("./Data/russtress/model.onnx");
            //sanity check
            {
                IReadOnlyDictionary<string, NodeMetadata> inputMetadata = this.m_session.InputMetadata;
                if (inputMetadata.Count != 1)
                {
                    throw new ApplicationException("Expected just a single input, but got " + inputMetadata.Count);
                }
                NodeMetadata inputNode = inputMetadata.First().Value;
                if (!inputNode.IsTensor)
                {
                    throw new ApplicationException("Expected input as tensor");
                }
                this.m_inputShape = inputNode.Dimensions;
                if (this.m_inputShape.Length != 3
                    || this.m_inputShape[0] != -1
                    || this.m_inputShape[1] != MAXLEN
                    || this.m_inputShape[2] != this.m_chars.Length)
                {
                    throw new ApplicationException($"got weird input shape [{String.Join(",", this.m_inputShape)}], expected [{1},{MAXLEN}, {this.m_chars.Length}]");
                }
                this.m_inputShape[0] = 1;   //specify batch size

                if (inputNode.ElementType != typeof(float))
                {
                    throw new ApplicationException($"got input element type '{inputNode.ElementType.Name}', while expected float");
                }
            }
            //sanity check
            {
                IReadOnlyDictionary<string, NodeMetadata> outputMetadata = this.m_session.OutputMetadata;
                if (outputMetadata.Count != 1)
                {
                    throw new ApplicationException("Expected just a single output, but got " + outputMetadata.Count);
                }
                NodeMetadata outputNode = outputMetadata.First().Value;
                if (!outputNode.IsTensor)
                {
                    throw new ApplicationException("Expected output as tensor");
                }
                int[] outputShape = outputNode.Dimensions;
                if (outputShape.Length != 2
                    || outputShape[0] != -1
                    || outputShape[1] != MAXLEN)
                {
                    throw new ApplicationException($"got weird output shape [{String.Join(",", outputShape)}], expected [{1},{MAXLEN}]");
                }

                if (outputNode.ElementType != typeof(float))
                {
                    throw new ApplicationException($"got output element type '{outputNode.ElementType.Name}', while expected float");
                }
            }
        }

        public void Dispose()
        {
            this.m_session.Dispose();
        }

        //a copy of text_accentAPI.py/__parse_the_phrase
        private List<string> ParseThePhrase(string text)
        {
            //text = text.Replace('c', 'с');  //latinic to cyrillic
            text = this.m_lineEndingRegex.Replace(text, " _ ").ToLower(); // mark beginning of clause
            // get rid of "#%&""()*-[0-9][a-z];=>@[\\]^_{|}\xa0'
            text = m_charsOnlyRegex.Replace(text, "");
            return text.Split(this.m_whitespaceSplit, StringSplitOptions.RemoveEmptyEntries).ToList();
        }

        //text_accentAPI.py/__add_endings
        private List<string> AddEndings(List<string> wordlist)
        {
            List<string> pluswords = new List<string>(wordlist.Count);
            for (int i = 0; i < wordlist.Count; ++i)
            {
                string word = wordlist[i];
                //if (!m_hasTwoVovels.IsMatch(word))  //won't predict, just return (less then two syllables )
                if (this.IsSmall(word))
                {
                    pluswords.Add(word);
                }
                else if (i == 0 || wordlist[i - 1] == "_")
                {
                    pluswords.Add('_' + word);
                }
                else
                {
                    string context = wordlist[i - 1].Replace("'", "");
                    string ending;
                    if (context.Length < 3)
                    {
                        ending = context;
                    }
                    else
                    {
                        ending = context.Substring(context.Length - 3);
                    }
                    string plusword = ending + '_' + word;
                    pluswords.Add(plusword);
                }
            }
            return pluswords;
        }

        private bool IsSmall(string word)
        {
            int count = 0;
            foreach (char c in word)
            {
                if (VOVELS.Contains(c))
                {
                    ++count;
                    if (count > 1)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        //tokenizer.py/tokenize
        private List<string> Tokenize(string str)
        {
            List<string> tokens = new List<string>();
            StringBuilder token = new StringBuilder();
            HashSet<char> category = null;
            foreach (char c in str)
            {
                if (token.Length > 0)
                {
                    if (category != null && category.Contains(c))
                    {
                        token.Append(c);
                    }
                    else
                    {
                        tokens.Add(token.ToString());
                        token = token.Clear().Append(c);
                        category = this.m_defaultCategories.FirstOrDefault(cat => cat.Contains(c));
                    }
                }
                else
                {
                    category = this.m_defaultCategories.FirstOrDefault(cat => cat.Contains(c));
                    token.Append(c);
                }
            }
            if (token.Length > 0)
            {
                tokens.Add(token.ToString());
            }
            return tokens;
        }

        //text_accentAPI.py/put_stress
        public string SetAccentForText(string text)
        {
            List<string> words = this.ParseThePhrase(text);
            List<string> pluswords = this.AddEndings(words);
            if (pluswords.Count != words.Count)
            {
                throw new ApplicationException($"lengths not match for '{text}'. Words: '{String.Join("|", words)}'. ContextedWords: '{String.Join("|", pluswords)}'");
            }

            List<string> accentedPhrase = new List<string>();
            for (int i = 0; i < words.Count; ++i)
            {
                string word = words[i];
                string contextedWord = pluswords[i];
                if (!contextedWord.EndsWith(word))
                {
                    throw new ApplicationException("Should not happen");
                }

                //if (!this.m_hasTwoVovels.IsMatch(word))   
                if (this.IsSmall(word)) //actually in original code they check contextedWord here, but it does not make sense for me
                {
                    continue;
                }
                else
                {
                    accentedPhrase.Add(this.PredictInternal(word, contextedWord));
                }
            }

            List<string> tokens = Tokenize(text);
            List<string> final = new List<string>();
            //TODO: this is very unsafe code considering the preprocessing done in the ParseThePhrase and AddEndings
            //copied from text_accentAPI.py/put_stress
            foreach (string token in tokens)
            {
                if (this.IsSmall(token))
                {
                    final.Add(token);
                }
                else if (accentedPhrase.Count > 0)
                {
                    string accentedWord = accentedPhrase.First();
                    string temp = accentedWord.Replace("'", "");
                    if (temp == token.ToLower())
                    {
                        int stressPosition = accentedWord.IndexOf('\'');
                        string accentedToken;
                        if (stressPosition >= 0)
                        {
                            accentedToken = token.Substring(0, stressPosition) + '\'' + token.Substring(stressPosition);
                        }
                        else
                        {
                            //no accent
                            accentedToken = token;
                        }
                        final.Add(accentedToken);
                        accentedPhrase.RemoveAt(0);
                    }
                    else
                    {
                        final.Add(token);
                    }
                }
            }
            return String.Join("", final);
        }

        //copy of text_accentAPI.py/__predict
        private string PredictInternal(string word, string wordWithContext)
        {
            if (wordWithContext.Length > MAXLEN)
            {
                return word; //no support for such long words
            }
            DenseTensor<float> tensor = new DenseTensor<float>(this.m_inputShape);
            for (int i = 0; i < wordWithContext.Length; ++i)
            {
                char letter = wordWithContext[i];
                int pos = MAXLEN - wordWithContext.Length + i;
                int charInd = this.m_charIndices[letter];
                tensor[0, pos, charInd] = 1.0f;
            }

            IReadOnlyCollection<NamedOnnxValue> inputs = new List<NamedOnnxValue>() {
                NamedOnnxValue.CreateFromTensor<float>(this.m_session.InputMetadata.First().Key, tensor)
            };
            float[] predictions;
            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = this.m_session.Run(inputs))
            {
                predictions = result.First().AsTensor<float>().ToArray();
            }

            int bestIndexInWord = -1;
            float bestProb = float.MinValue;

            int checkIndexInProb = predictions.Length - 1;
            int checkIndexInWord = word.Length - 1;
            while (checkIndexInWord >= 0)
            {
                if (VOVELS.Contains(word[checkIndexInWord]))
                {
                    if (predictions[checkIndexInProb] > bestProb)
                    {
                        bestProb = predictions[checkIndexInProb];
                        bestIndexInWord = checkIndexInWord;
                    }
                }
                --checkIndexInWord;
                --checkIndexInProb;
            }
            if (bestIndexInWord < 0)
            {
                return word;
            }
            return word.Substring(0, bestIndexInWord + 1) + '\'' + word.Substring(bestIndexInWord + 1);
        }

        private static HashSet<char> StringToHashSet(string s)
        {
            HashSet<char> result = new HashSet<char>();
            foreach (char c in s)
            {
                result.Add(c);
            }
            return result;
        }
    }
}
