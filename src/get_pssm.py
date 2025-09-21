#!/usr/bin/env python
# Author  : KerryChen
# File    : runBlast_SNAP.py
# Time    : 2024/10/25 15:51


import os
import subprocess


class PsiBlast():
    def runBlast(self, fastapath, output_dir):
        names = [name for name in os.listdir(fastapath) if os.path.isfile(os.path.join(fastapath + '//', name))]

        tool_path = '/home/software/blast/bin/psiblast'
        db_path = '/home/software/blast/database/swissprot'
        evalue = 0.001
        num_iterations = 3

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for each_item in names:
            pdb_id = each_item.split('.')[0]
            postfix = each_item.split('.')[1]

            if postfix == 'fasta':
                output_file = os.path.join(output_dir, pdb_id + '.pssm')
                fasta_file = fastapath + '/' + each_item

                if os.path.exists(output_file):
                    print(f'Skipping {fasta_file} because {output_file} already exists.')
                    continue 
                # cmd = f'{tool_path} -query {fasta_file} -db {db_path} -evalue 0.001 -num_iterations 3 -out_ascii_pssm {output_file}'
                # print(cmd)
                # os.system(cmd)

                command = [
                    tool_path,
                    '-query', fasta_file,
                    '-db', db_path,
                    '-evalue', str(evalue),
                    '-num_iterations', str(num_iterations),
                    '-out_ascii_pssm', output_file
                ]


                try:
                    subprocess.run(command, check=True)
                    print(f'Successfully ran psiblast for {fasta_file}')
                except subprocess.CalledProcessError as e:
                    print(f'Failed to run psiblast for {fasta_file}: {e}')


if __name__ == '__main__':
    fastapath = '../fasta'
    outdir = '../pssm'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pb = PsiBlast()
    pb.runBlast(fastapath, outdir)


